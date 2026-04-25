"""
rag_retriever.py — Retrieval-Augmented Generation Layer

This module implements a lightweight RAG pipeline that lets users query songs
using natural language (e.g., "something for a late night drive") instead of
explicit feature values.

Architecture:
  1. INDEXING: Each song is converted into a rich text description embedding
     its genre, mood, energy level, and audio features as semantic text.
  2. RETRIEVAL: At query time, TF-IDF cosine similarity finds songs whose
     descriptions best match the natural language query.
  3. AUGMENTATION: The top-K retrieved songs are passed to the recommender
     as pre-filtered candidates, improving relevance for vague queries.

Why TF-IDF instead of neural embeddings?
  - Zero external API calls → works fully offline
  - Deterministic and debuggable
  - Fast enough for 5k+ songs without GPU
  - Interpretable: you can see exactly which words matched
  - Easily swappable: the interface is compatible with OpenAI/Cohere embeddings
"""

import logging
import math
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Song descriptor vocabulary ───────────────────────────────────────────────

# Maps audio feature ranges to human-readable descriptors.
# This is the "document" that gets indexed and searched.

ENERGY_DESCRIPTORS = [
    (0.80, ["high energy", "intense", "powerful", "driving", "hard", "aggressive",
            "workout", "gym", "running", "hype", "electric", "loud"]),
    (0.55, ["moderate energy", "upbeat", "lively", "active", "dynamic"]),
    (0.35, ["low energy", "calm", "peaceful", "gentle", "soft", "quiet",
            "relaxed", "background", "study", "focus", "mellow"]),
    (0.00, ["very low energy", "ambient", "minimal", "still", "drone",
            "sleep", "meditation", "whisper"]),
]

VALENCE_DESCRIPTORS = [
    (0.70, ["happy", "joyful", "uplifting", "cheerful", "positive", "feel good",
            "fun", "bright", "euphoric", "celebratory", "good vibes"]),
    (0.45, ["bittersweet", "nostalgic", "wistful", "neutral", "melancholic"]),
    (0.00, ["sad", "dark", "heavy", "gloomy", "melancholy", "emotional",
            "heartbreak", "deep", "brooding", "somber", "tragic"]),
]

TEMPO_DESCRIPTORS = [
    (160, ["very fast", "frantic", "breakneck", "sprint"]),
    (130, ["fast", "uptempo", "driving beat", "energetic tempo"]),
    (100, ["moderate tempo", "mid tempo", "steady"]),
    (75,  ["slow", "slow tempo", "laid back"]),
    (0,   ["very slow", "glacial", "crawling", "downtempo"]),
]

DANCEABILITY_DESCRIPTORS = [
    (0.75, ["danceable", "dance", "groove", "club", "party", "floor filler"]),
    (0.50, ["moderately danceable", "rhythmic", "groovy"]),
    (0.00, ["not danceable", "listening music", "concert", "headphone"]),
]

ACOUSTICNESS_DESCRIPTORS = [
    (0.70, ["acoustic", "unplugged", "organic", "live instruments",
            "raw", "stripped", "natural", "guitar", "piano"]),
    (0.30, ["semi-acoustic", "hybrid"]),
    (0.00, ["electronic", "produced", "synthesized", "digital", "studio"]),
]

MOOD_SYNONYMS = {
    "happy":   ["happy", "joyful", "cheerful", "upbeat", "feel good", "positive",
                "bright", "euphoric", "energetic", "fun"],
    "intense": ["intense", "aggressive", "powerful", "hard", "angry", "fierce",
                "dark energy", "metal", "heavy", "driving", "gritty"],
    "relaxed": ["relaxed", "chill", "peaceful", "easy", "laid back", "calm",
                "breezy", "light", "smooth", "comfortable"],
    "chill":   ["chill", "lo-fi", "lofi", "study", "coffee shop", "background",
                "night", "late night", "cozy", "warm", "sleepy", "quiet"],
    "moody":   ["moody", "dark", "atmospheric", "brooding", "complex", "deep",
                "emotional", "introspective", "rainy", "dreamy"],
    "focused": ["focused", "concentration", "work", "productivity", "flow",
                "clean", "minimal", "instrumental", "steady"],
}

# Context/scene keywords that map to song characteristics
CONTEXT_KEYWORDS = {
    "late night drive":    {"energy": 0.6, "valence": 0.4, "mood": "moody"},
    "morning run":         {"energy": 0.85, "valence": 0.7, "mood": "happy"},
    "study session":       {"energy": 0.3, "valence": 0.5, "mood": "focused"},
    "party":               {"energy": 0.85, "valence": 0.8, "mood": "happy"},
    "heartbreak":          {"energy": 0.4, "valence": 0.2, "mood": "moody"},
    "gym workout":         {"energy": 0.9, "valence": 0.6, "mood": "intense"},
    "road trip":           {"energy": 0.75, "valence": 0.7, "mood": "happy"},
    "meditation":          {"energy": 0.1, "valence": 0.5, "mood": "chill"},
    "rainy day":           {"energy": 0.3, "valence": 0.35, "mood": "chill"},
    "coffee shop":         {"energy": 0.35, "valence": 0.6, "mood": "relaxed"},
    "pregame":             {"energy": 0.88, "valence": 0.75, "mood": "happy"},
    "coding":              {"energy": 0.45, "valence": 0.5, "mood": "focused"},
    "sunday morning":      {"energy": 0.25, "valence": 0.65, "mood": "relaxed"},
    "sadness":             {"energy": 0.35, "valence": 0.15, "mood": "moody"},
}


def _get_descriptor(value: float, table: list) -> List[str]:
    for threshold, words in table:
        if value >= threshold:
            return words
    return table[-1][1]


def build_song_document(song) -> str:
    """
    Converts a Song's audio features into a rich natural-language text document
    that can be indexed and searched semantically.
    """
    parts = []

    # Core metadata
    parts.append(song.genre.replace("-", " ").replace("_", " "))
    parts.append(song.mood)
    parts.extend(MOOD_SYNONYMS.get(song.mood, [song.mood]))

    # Audio feature descriptors
    parts.extend(_get_descriptor(song.energy, ENERGY_DESCRIPTORS))
    parts.extend(_get_descriptor(song.valence, VALENCE_DESCRIPTORS))
    parts.extend(_get_descriptor(song.tempo_bpm, TEMPO_DESCRIPTORS))
    parts.extend(_get_descriptor(song.danceability, DANCEABILITY_DESCRIPTORS))
    parts.extend(_get_descriptor(song.acousticness, ACOUSTICNESS_DESCRIPTORS))

    # Artist name (helps with "something like Drake" queries)
    parts.append(song.artist.lower())

    return " ".join(parts)


# ── TF-IDF Index ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', text.lower())


class TFIDFIndex:
    """
    Lightweight TF-IDF index for semantic song retrieval.
    Built from scratch — no scikit-learn dependency required.
    """

    def __init__(self):
        self.songs = []
        self.documents = []
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: List[Dict[str, float]] = []

    def build(self, songs: list) -> None:
        """Index all songs. Call once at startup."""
        logger.info(f"Building TF-IDF index for {len(songs)} songs...")
        self.songs = songs
        self.documents = [_tokenize(build_song_document(s)) for s in songs]

        # Compute IDF
        N = len(self.documents)
        df: Dict[str, int] = defaultdict(int)
        for doc in self.documents:
            for term in set(doc):
                df[term] += 1
        self.idf = {term: math.log((N + 1) / (freq + 1)) + 1
                    for term, freq in df.items()}

        # Compute TF-IDF vectors
        self.tfidf_vectors = []
        for doc in self.documents:
            tf: Dict[str, float] = defaultdict(float)
            for term in doc:
                tf[term] += 1
            length = len(doc) or 1
            vec = {term: (count / length) * self.idf.get(term, 1.0)
                   for term, count in tf.items()}
            self.tfidf_vectors.append(vec)

        logger.info("TF-IDF index built.")

    def _cosine_sim(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        dot = sum(vec_a[t] * vec_b[t] for t in common)
        mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
        mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def query(self, text: str, k: int = 20) -> List[Tuple[object, float]]:
        """Returns top-k (song, similarity_score) pairs for a natural language query."""
        tokens = _tokenize(text)
        length = len(tokens) or 1
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        query_vec = {t: (count / length) * self.idf.get(t, 1.0)
                     for t, count in tf.items()}

        scored = [
            (self.songs[i], self._cosine_sim(query_vec, self.tfidf_vectors[i]))
            for i in range(len(self.songs))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ── RAG Retriever ─────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    songs: list
    query: str
    expanded_query: str
    method: str          # "semantic" | "context" | "hybrid"
    top_scores: List[float]
    confidence: float


class RAGRetriever:
    """
    Retrieval-Augmented Generation layer for the music recommender.

    Workflow:
      1. Query expansion: detect context keywords, expand with synonyms
      2. Semantic retrieval: TF-IDF cosine similarity search
      3. Result augmentation: retrieved songs become the candidate pool
         fed into the scoring recommender
    """

    def __init__(self):
        self.index = TFIDFIndex()
        self._built = False

    def build_index(self, songs: list) -> None:
        self.index.build(songs)
        self._built = True

    def expand_query(self, query: str) -> Tuple[str, str]:
        """
        Expands a user query with synonyms and context mappings.
        Returns (expanded_query, method_used).
        """
        q_lower = query.lower()
        expansion_parts = [query]
        method = "semantic"

        # Check for context keywords
        for context, features in CONTEXT_KEYWORDS.items():
            if any(word in q_lower for word in context.split()):
                mood = features.get("mood", "")
                expansion_parts.extend(MOOD_SYNONYMS.get(mood, [mood]))
                method = "context"
                logger.debug(f"Context match: '{context}' → mood='{mood}'")
                break

        # Expand mood synonyms
        for mood, synonyms in MOOD_SYNONYMS.items():
            if mood in q_lower or any(s in q_lower for s in synonyms[:3]):
                expansion_parts.extend(synonyms)
                break

        expanded = " ".join(expansion_parts)
        return expanded, method

    def retrieve(self, query: str, k: int = 30) -> RetrievalResult:
        """
        Main retrieval method. Returns a RetrievalResult with candidate songs.

        Args:
            query: Natural language query (e.g., "chill beats for studying")
            k:     Number of candidates to retrieve before final scoring
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build_index(songs) first.")

        expanded_query, method = self.expand_query(query)
        logger.info(f"RAG query: '{query}' → expanded: '{expanded_query[:80]}...'")

        results = self.index.query(expanded_query, k=k)
        songs = [r[0] for r in results]
        scores = [r[1] for r in results]

        # Confidence: ratio of top score to mean score (measures result clarity)
        if scores:
            top = scores[0]
            mean = sum(scores) / len(scores) if scores else 0
            confidence = round(min(1.0, top / (mean + 1e-9) / 3.0), 3)
        else:
            confidence = 0.0

        logger.info(f"Retrieved {len(songs)} candidates. Confidence: {confidence:.3f}")

        return RetrievalResult(
            songs=songs,
            query=query,
            expanded_query=expanded_query,
            method=method,
            top_scores=scores[:5],
            confidence=confidence,
        )

    def retrieve_for_profile(self, user_prefs: Dict, k: int = 30) -> RetrievalResult:
        """
        Builds a natural language query from a structured user profile,
        then retrieves candidates. Bridges the old dict-based API with RAG.
        """
        parts = []
        genre = user_prefs.get("favorite_genre", "")
        mood  = user_prefs.get("favorite_mood", "")
        energy = user_prefs.get("target_energy", 0.5)

        if genre:
            parts.append(genre)
        if mood:
            parts.append(mood)
            parts.extend(MOOD_SYNONYMS.get(mood, [])[:3])
        if energy >= 0.7:
            parts.extend(["high energy", "upbeat", "driving"])
        elif energy <= 0.35:
            parts.extend(["low energy", "calm", "chill", "relaxed"])

        query = " ".join(parts) if parts else "music"
        return self.retrieve(query, k=k)
