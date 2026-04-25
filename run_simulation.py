"""
run_simulation.py — Main CLI Runner for the Applied AI Music Recommender

Modes:
  python run_simulation.py              # Classic profile-based demo
  python run_simulation.py --mode agent # Natural language agent demo
  python run_simulation.py --mode both  # Both demos
  python run_simulation.py --query "chill beats for studying"
"""

import argparse
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.guardrails import setup_logging, validate_user_prefs, validate_query, print_validation_summary
from src.data_loader import load_songs, songs_to_dicts
from src.recommender import recommend_songs
from src.rag_retriever import RAGRetriever
from src.agent import MusicRecommenderAgent


# ── Classic profile demo (original run_simulation behavior) ─────────────────

def run_classic_demo(songs, verbose: bool = False):
    song_dicts = songs_to_dicts(songs)

    test_profiles = {
        "High-Energy Pop":               {"favorite_genre": "pop",       "favorite_mood": "happy",   "target_energy": 0.90},
        "Chill Lo-Fi":                   {"favorite_genre": "lo-fi",     "favorite_mood": "chill",   "target_energy": 0.30},
        "Deep Intense Rock":             {"favorite_genre": "rock",      "favorite_mood": "intense", "target_energy": 0.85},
        "Late Night R&B":                {"favorite_genre": "r&b",       "favorite_mood": "moody",   "target_energy": 0.50},
        "The Conflicted User (Adversarial)": {"favorite_genre": "ambient", "favorite_mood": "intense", "target_energy": 0.95},
    }

    print(f"\n{'═'*55}")
    print(f"  🎵  MUSIC RECOMMENDER — Classic Profile Mode")
    print(f"{'═'*55}")

    for profile_name, raw_prefs in test_profiles.items():
        # Run through guardrails
        val = validate_user_prefs(raw_prefs)
        print_validation_summary(val, context=profile_name)
        user_prefs = val.value

        print(f"\n{'─'*45}")
        print(f"  Profile: {profile_name}")
        print(f"  Genre: {user_prefs['favorite_genre']} | Mood: {user_prefs['favorite_mood']} | Energy: {user_prefs['target_energy']}")
        print(f"{'─'*45}")

        recommendations = recommend_songs(user_prefs, song_dicts, k=3)

        for rec in recommendations:
            song, score, explanation = rec
            print(f"  ♪  {song['title']} — {song['artist']}")
            print(f"     Score: {score:.2f} | {explanation}")
            print()


# ── Agentic demo ─────────────────────────────────────────────────────────────

def run_agent_demo(songs, retriever, queries=None):
    agent = MusicRecommenderAgent(songs, retriever)

    default_queries = [
        "chill lo-fi beats for a late night study session",
        "high energy hip hop to get hyped for the gym",
        "something moody and atmospheric for a rainy Sunday drive",
        "happy upbeat pop for a morning run",
        "deep jazz for a quiet coffee shop afternoon",
    ]
    queries = queries or default_queries

    print(f"\n{'═'*55}")
    print(f"  🤖  MUSIC RECOMMENDER — Agentic Mode (RAG + Planning)")
    print(f"{'═'*55}")

    for query in queries:
        result = agent.run(query, k=5, verbose=True)

        print(f"\n  🎵 Recommendations for: '{query}'")
        print(f"  {'─'*48}")
        for i, (song, expl) in enumerate(zip(result.recommendations, result.explanations), 1):
            print(f"  {i}. {song.title} — {song.artist}")
            print(f"     [{song.genre} / {song.mood} / energy {song.energy:.2f}]")
            print(f"     Why: {expl}")
        print(f"\n  Confidence: {result.confidence:.2f} | "
              f"Steps: {len(result.steps)} | "
              f"Time: {result.total_duration_ms:.0f}ms | "
              f"Retried: {result.retried}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Music Recommender System")
    parser.add_argument("--mode", choices=["classic", "agent", "both"],
                        default="both", help="Which demo to run")
    parser.add_argument("--query", type=str, help="Run a single agent query")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--data", type=str, help="Path to Kaggle CSV", default=None)
    args = parser.parse_args()

    setup_logging(args.log_level)

    print("\n  Loading song catalog...")
    songs = load_songs(args.data)
    print(f"  ✅ Loaded {len(songs)} songs")

    print("  Building RAG index...")
    retriever = RAGRetriever()
    retriever.build_index(songs)
    print("  ✅ Index ready\n")

    if args.query:
        val = validate_query(args.query)
        print_validation_summary(val, "query")
        if not val.is_valid:
            print(f"Invalid query: {val.warnings}")
            return
        agent = MusicRecommenderAgent(songs, retriever)
        result = agent.run(val.value, k=5, verbose=True)
        print(f"\n🎵 Results for: '{args.query}'")
        for i, (s, e) in enumerate(zip(result.recommendations, result.explanations), 1):
            print(f"  {i}. {s.title} — {s.artist}  ({s.genre} / {s.mood})")
            print(f"     {e}")
        return

    if args.mode in ("classic", "both"):
        run_classic_demo(songs)

    if args.mode in ("agent", "both"):
        run_agent_demo(songs, retriever)


if __name__ == "__main__":
    main()
