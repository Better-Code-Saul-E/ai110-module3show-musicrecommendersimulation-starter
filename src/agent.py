"""
agent.py — Agentic Recommendation Workflow

This module implements a multi-step reasoning agent that:
  1. UNDERSTANDS the user's intent (natural language → structured preferences)
  2. PLANS which tools to invoke (RAG retrieval, scoring, filtering)
  3. ACTS by calling tools in sequence with observable intermediate steps
  4. EVALUATES its own output (confidence scoring, diversity check)
  5. REFLECTS and optionally retries if quality is too low

This satisfies the "Agentic Workflow Enhancement" stretch requirement by
implementing observable intermediate steps and a decision-making chain.

Observable steps are logged to: logs/agent_trace.jsonl
"""

import json
import logging
import time
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Agent state and trace ────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """A single observable step in the agent's reasoning chain."""
    step_num: int
    tool: str
    input: Dict
    output_summary: str
    duration_ms: float
    success: bool
    notes: str = ""


@dataclass
class AgentResult:
    """Full result from one agent run, including all intermediate steps."""
    query: str
    resolved_preferences: Dict
    recommendations: List[Any]       # List of Song
    explanations: List[str]
    confidence: float
    steps: List[AgentStep] = field(default_factory=list)
    retried: bool = False
    total_duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Tool definitions ─────────────────────────────────────────────────────────

class AgentTools:
    """
    The set of tools the agent can call. Each tool is a discrete,
    observable action with logged inputs and outputs.
    """

    def __init__(self, songs, retriever, recommender):
        self.songs = songs
        self.retriever = retriever
        self.recommender = recommender

    def tool_parse_intent(self, query: str) -> Dict:
        """
        Tool: parse_intent
        Converts natural language query into structured preference dict.
        Uses keyword matching + heuristics (no LLM call needed for this).
        """
        q = query.lower()
        prefs = {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.65,
            "raw_query": query,
            "source": "parsed",
        }

        # Genre detection
        genre_keywords = {
            "hip-hop": ["hip hop", "hiphop", "rap", "trap", "drill"],
            "rock": ["rock", "guitar", "metal", "punk", "grunge", "indie rock"],
            "electronic": ["electronic", "edm", "techno", "house", "rave", "synth"],
            "jazz": ["jazz", "blues", "swing", "bebop", "soul jazz"],
            "lo-fi": ["lofi", "lo-fi", "lo fi", "study beats", "chill beats"],
            "r&b": ["r&b", "rnb", "soul", "neo soul"],
            "ambient": ["ambient", "drone", "atmospheric", "meditation"],
            "classical": ["classical", "orchestra", "piano", "violin", "symphony"],
            "latin": ["latin", "reggaeton", "salsa", "cumbia", "bachata"],
            "metal": ["metal", "heavy", "thrash", "death metal"],
            "indie": ["indie", "alternative", "shoegaze", "dream pop"],
        }
        for genre, keywords in genre_keywords.items():
            if any(kw in q for kw in keywords):
                prefs["favorite_genre"] = genre
                break

        # Mood + energy detection
        mood_energy_map = {
            "happy":   ("happy",   0.78),
            "intense": ("intense", 0.88),
            "chill":   ("chill",   0.30),
            "relaxed": ("relaxed", 0.45),
            "moody":   ("moody",   0.42),
            "focused": ("focused", 0.40),
            "sad":     ("moody",   0.35),
            "angry":   ("intense", 0.90),
            "energetic": ("happy", 0.85),
            "hype":    ("intense", 0.92),
            "calm":    ("chill",   0.25),
            "workout": ("intense", 0.90),
            "study":   ("focused", 0.35),
            "sleep":   ("chill",   0.15),
            "drive":   ("moody",   0.65),
            "party":   ("happy",   0.85),
        }
        for keyword, (mood, energy) in mood_energy_map.items():
            if keyword in q:
                prefs["favorite_mood"] = mood
                prefs["target_energy"] = energy
                break

        # Energy override from explicit descriptors
        if any(w in q for w in ["loud", "hard", "fast", "aggressive", "powerful"]):
            prefs["target_energy"] = max(prefs["target_energy"], 0.80)
        if any(w in q for w in ["soft", "quiet", "gentle", "slow", "easy"]):
            prefs["target_energy"] = min(prefs["target_energy"], 0.40)

        return prefs

    def tool_rag_retrieve(self, query: str, k: int = 40) -> Tuple[List, float]:
        """
        Tool: rag_retrieve
        Uses the RAG index to retrieve semantically relevant candidates.
        Returns (candidate_songs, retrieval_confidence).
        """
        result = self.retriever.retrieve(query, k=k)
        return result.songs, result.confidence

    def tool_score_and_rank(self, candidates: List, prefs: Dict, k: int = 5) -> List:
        """
        Tool: score_and_rank
        Applies the scoring algorithm to RAG candidates to get final top-k.
        """
        from src.recommender import Recommender, UserProfile
        rec = Recommender(candidates)
        user = UserProfile(
            favorite_genre=prefs["favorite_genre"],
            favorite_mood=prefs["favorite_mood"],
            target_energy=prefs["target_energy"],
            likes_acoustic=prefs.get("target_energy", 0.5) < 0.45,
        )
        return rec.recommend(user, k=k)

    def tool_check_diversity(self, songs: List) -> Tuple[bool, str]:
        """
        Tool: check_diversity
        Evaluates whether recommendations are diverse enough.
        Returns (is_diverse, reason).
        """
        if not songs:
            return False, "empty result set"

        genres = [s.genre for s in songs]
        artists = [s.artist for s in songs]
        moods = [s.mood for s in songs]

        genre_diversity  = len(set(genres)) / len(genres)
        artist_diversity = len(set(artists)) / len(artists)

        if genre_diversity < 0.3:
            return False, f"low genre diversity ({len(set(genres))} genres for {len(songs)} songs)"
        if artist_diversity < 0.5:
            return False, f"too many songs from same artist"

        mood_counts = {m: moods.count(m) for m in set(moods)}
        dominant_mood_pct = max(mood_counts.values()) / len(moods)
        if dominant_mood_pct > 0.8:
            return True, f"acceptable (mood slightly uniform at {dominant_mood_pct:.0%})"

        return True, f"good diversity — {len(set(genres))} genres, {len(set(artists))} artists"

    def tool_compute_confidence(self, songs: List, prefs: Dict,
                                 retrieval_conf: float) -> float:
        """
        Tool: compute_confidence
        Computes an overall confidence score for the recommendation set.
        Components:
          - Genre match rate (0–1)
          - Mood match rate (0–1)
          - Energy proximity (0–1)
          - Retrieval confidence from RAG (0–1)
        """
        if not songs:
            return 0.0

        genre_hits = sum(1 for s in songs if s.genre == prefs["favorite_genre"]) / len(songs)
        mood_hits  = sum(1 for s in songs if s.mood  == prefs["favorite_mood"])  / len(songs)
        energy_prox = sum(
            max(0.0, 1.0 - abs(s.energy - prefs["target_energy"]))
            for s in songs
        ) / len(songs)

        confidence = (
            genre_hits   * 0.30 +
            mood_hits    * 0.25 +
            energy_prox  * 0.25 +
            retrieval_conf * 0.20
        )
        return round(min(1.0, confidence), 3)

    def tool_explain(self, song, prefs: Dict) -> str:
        """
        Tool: explain
        Generates a human-readable explanation for why a song was recommended.
        """
        reasons = []
        if song.genre == prefs["favorite_genre"]:
            reasons.append(f"matches your {song.genre} preference")
        if song.mood == prefs["favorite_mood"]:
            reasons.append(f"{song.mood} mood")
        energy_gap = abs(song.energy - prefs["target_energy"])
        if energy_gap < 0.15:
            reasons.append("energy level spot-on")
        elif energy_gap < 0.30:
            reasons.append("close energy match")
        if song.acousticness > 0.6:
            reasons.append("acoustic texture")
        if song.danceability > 0.75:
            reasons.append("highly danceable")
        return ", ".join(reasons) if reasons else "general match"


# ── The Agent ─────────────────────────────────────────────────────────────────

class MusicRecommenderAgent:
    """
    Multi-step reasoning agent for music recommendation.

    Decision chain:
      parse_intent → rag_retrieve → score_and_rank →
      check_diversity → compute_confidence →
      [retry if confidence < threshold] → explain → return
    """

    CONFIDENCE_THRESHOLD = 0.45  # Retry if below this
    TRACE_LOG_PATH = "logs/agent_trace.jsonl"

    def __init__(self, songs, retriever, recommender=None):
        self.tools = AgentTools(songs, retriever, recommender)
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        os.makedirs(os.path.dirname(self.TRACE_LOG_PATH), exist_ok=True)

    def _log_step(self, step: AgentStep, result: AgentResult):
        result.steps.append(step)
        status = "✓" if step.success else "✗"
        logger.info(f"  [{status}] Step {step.step_num}: {step.tool} "
                    f"({step.duration_ms:.0f}ms) — {step.output_summary}")

    def _timed_call(self, fn, *args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        return out, round((time.time() - t0) * 1000, 1)

    def run(self, query: str, k: int = 5, verbose: bool = True) -> AgentResult:
        """
        Run the full agentic pipeline for a given natural language query.

        Args:
            query:   Natural language query (e.g. "chill lo-fi for studying")
            k:       Number of recommendations to return
            verbose: Print intermediate steps to console

        Returns:
            AgentResult with recommendations and full reasoning trace
        """
        t_start = time.time()
        result = AgentResult(query=query, resolved_preferences={},
                             recommendations=[], explanations=[], confidence=0.0)
        step_num = 0

        if verbose:
            print(f"\n🤖 Agent starting for query: '{query}'")
            print("─" * 50)

        # ── Step 1: Parse Intent ──────────────────────────────────────────
        step_num += 1
        prefs, ms = self._timed_call(self.tools.tool_parse_intent, query)
        result.resolved_preferences = prefs
        self._log_step(AgentStep(
            step_num=step_num, tool="parse_intent",
            input={"query": query},
            output_summary=f"genre={prefs['favorite_genre']}, mood={prefs['favorite_mood']}, energy={prefs['target_energy']}",
            duration_ms=ms, success=True,
        ), result)
        if verbose:
            print(f"  📋 Intent: genre={prefs['favorite_genre']}, mood={prefs['favorite_mood']}, energy={prefs['target_energy']}")

        # ── Step 2: RAG Retrieval ─────────────────────────────────────────
        step_num += 1
        rag_query = f"{prefs['favorite_genre']} {prefs['favorite_mood']} {query}"
        (candidates, retrieval_conf), ms = self._timed_call(
            self.tools.tool_rag_retrieve, rag_query, k * 8
        )
        self._log_step(AgentStep(
            step_num=step_num, tool="rag_retrieve",
            input={"query": rag_query, "k": k * 8},
            output_summary=f"retrieved {len(candidates)} candidates (conf={retrieval_conf:.3f})",
            duration_ms=ms, success=len(candidates) > 0,
        ), result)
        if verbose:
            print(f"  🔍 RAG: {len(candidates)} candidates retrieved (confidence={retrieval_conf:.2f})")

        # ── Step 3: Score & Rank ──────────────────────────────────────────
        step_num += 1
        ranked, ms = self._timed_call(
            self.tools.tool_score_and_rank, candidates, prefs, k
        )
        self._log_step(AgentStep(
            step_num=step_num, tool="score_and_rank",
            input={"candidates": len(candidates), "k": k},
            output_summary=f"top song: '{ranked[0].title if ranked else 'none'}' by {ranked[0].artist if ranked else '?'}",
            duration_ms=ms, success=len(ranked) > 0,
        ), result)
        if verbose and ranked:
            print(f"  🎵 Top pick: '{ranked[0].title}' by {ranked[0].artist}")

        # ── Step 4: Diversity Check ───────────────────────────────────────
        step_num += 1
        (is_diverse, diversity_reason), ms = self._timed_call(
            self.tools.tool_check_diversity, ranked
        )
        self._log_step(AgentStep(
            step_num=step_num, tool="check_diversity",
            input={"n_songs": len(ranked)},
            output_summary=diversity_reason,
            duration_ms=ms, success=is_diverse,
            notes="diversity acceptable" if is_diverse else "low diversity detected",
        ), result)
        if verbose:
            icon = "✅" if is_diverse else "⚠️"
            print(f"  {icon} Diversity: {diversity_reason}")

        # ── Step 5: Confidence Scoring ────────────────────────────────────
        step_num += 1
        confidence, ms = self._timed_call(
            self.tools.tool_compute_confidence, ranked, prefs, retrieval_conf
        )
        self._log_step(AgentStep(
            step_num=step_num, tool="compute_confidence",
            input={"n_songs": len(ranked)},
            output_summary=f"confidence={confidence:.3f} (threshold={self.CONFIDENCE_THRESHOLD})",
            duration_ms=ms, success=confidence >= self.CONFIDENCE_THRESHOLD,
        ), result)
        if verbose:
            bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"  📊 Confidence: [{bar}] {confidence:.2f}")

        # ── Step 6: Retry if low confidence ──────────────────────────────
        if confidence < self.CONFIDENCE_THRESHOLD and not result.retried:
            step_num += 1
            if verbose:
                print(f"  🔄 Confidence below {self.CONFIDENCE_THRESHOLD} — retrying with broader search...")
            result.retried = True

            # Broaden search: relax genre constraint, increase candidate pool
            broad_prefs = dict(prefs)
            broad_prefs["favorite_genre"] = ""  # Remove genre filter
            broad_query = f"{prefs['favorite_mood']} {query} music"
            (candidates2, retrieval_conf2), ms2 = self._timed_call(
                self.tools.tool_rag_retrieve, broad_query, k * 15
            )
            ranked2, _ = self._timed_call(
                self.tools.tool_score_and_rank, candidates2, broad_prefs, k
            )
            confidence2, _ = self._timed_call(
                self.tools.tool_compute_confidence, ranked2, prefs, retrieval_conf2
            )

            if confidence2 > confidence:
                ranked = ranked2
                confidence = confidence2
                self._log_step(AgentStep(
                    step_num=step_num, tool="retry_broad",
                    input={"broad_query": broad_query},
                    output_summary=f"retry improved confidence: {confidence:.3f}",
                    duration_ms=ms2, success=True,
                ), result)
                if verbose:
                    print(f"  ✅ Retry succeeded: confidence now {confidence:.2f}")
            else:
                self._log_step(AgentStep(
                    step_num=step_num, tool="retry_broad",
                    input={"broad_query": broad_query},
                    output_summary=f"retry did not improve ({confidence2:.3f} vs {confidence:.3f}), keeping original",
                    duration_ms=ms2, success=False,
                ), result)
                if verbose:
                    print(f"  ℹ️  Keeping original results.")

        # ── Step 7: Generate Explanations ─────────────────────────────────
        step_num += 1
        explanations = []
        for song in ranked:
            expl, ms = self._timed_call(self.tools.tool_explain, song, prefs)
            explanations.append(expl)
        self._log_step(AgentStep(
            step_num=step_num, tool="explain",
            input={"n_songs": len(ranked)},
            output_summary=f"generated {len(explanations)} explanations",
            duration_ms=ms, success=True,
        ), result)

        # ── Finalize ──────────────────────────────────────────────────────
        result.recommendations = ranked
        result.explanations = explanations
        result.confidence = confidence
        result.total_duration_ms = round((time.time() - t_start) * 1000, 1)

        if verbose:
            print(f"─" * 50)
            print(f"  ⏱️  Total: {result.total_duration_ms:.0f}ms | "
                  f"Confidence: {confidence:.2f} | Steps: {len(result.steps)}")

        self._write_trace(result)
        return result

    def _write_trace(self, result: AgentResult):
        """Appends agent trace to JSONL log for analysis."""
        try:
            trace = {
                "timestamp": result.timestamp,
                "query": result.query,
                "confidence": result.confidence,
                "retried": result.retried,
                "n_results": len(result.recommendations),
                "total_ms": result.total_duration_ms,
                "steps": [
                    {"tool": s.tool, "success": s.success, "ms": s.duration_ms,
                     "output": s.output_summary}
                    for s in result.steps
                ],
            }
            with open(self.TRACE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write trace: {e}")
