"""
recommender.py — Core Recommendation Logic

This file is the original Module 1-3 implementation, preserved and extended.
The Song and UserProfile dataclasses are kept as the canonical schema.
Scoring logic is unchanged for test compatibility.

New in this version:
  - Song and UserProfile now imported from data_loader (single source of truth)
  - Added confidence scoring to Recommender
  - recommend_songs() dict-based API preserved for backward compatibility
"""

import csv
import logging
from typing import List, Dict, Tuple, Optional

# Re-export from data_loader so existing imports still work
from src.data_loader import Song, UserProfile, load_songs, songs_to_dicts

logger = logging.getLogger(__name__)


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Compatible with tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored_songs = []

        for song in self.songs:
            score = 0.0

            # 1. Genre Match (+1.0)
            if song.genre.lower() == user.favorite_genre.lower():
                score += 1.0

            # 2. Mood Match (+1.0)
            if song.mood.lower() == user.favorite_mood.lower():
                score += 1.0

            # 3. Energy Gap (up to +2.0)
            energy_gap = abs(song.energy - user.target_energy)
            score += max(0.0, 1.0 - energy_gap) * 2.0

            scored_songs.append((song, score))

        scored_songs.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_songs[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        reasons = []
        if song.genre.lower() == user.favorite_genre.lower():
            reasons.append("genre match")
        if song.mood.lower() == user.favorite_mood.lower():
            reasons.append("mood match")
        energy_gap = abs(song.energy - user.target_energy)
        energy_points = max(0.0, 1.0 - energy_gap) * 2.0
        reasons.append(f"energy match (+{energy_points:.2f})")
        return ", ".join(reasons)

    def recommend_with_confidence(self, user: UserProfile, k: int = 5) -> Tuple[List[Song], float]:
        """Extended version that also returns a confidence score."""
        results = self.recommend(user, k)
        if not results:
            return [], 0.0

        genre_hits = sum(1 for s in results if s.genre == user.favorite_genre.lower()) / len(results)
        mood_hits  = sum(1 for s in results if s.mood  == user.favorite_mood.lower())  / len(results)
        energy_prox = sum(
            max(0.0, 1.0 - abs(s.energy - user.target_energy))
            for s in results
        ) / len(results)

        confidence = round((genre_hits * 0.4 + mood_hits * 0.3 + energy_prox * 0.3), 3)
        return results, confidence


# ── Dict-based API (backward compat with run_simulation.py) ─────────────────

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song dict against user preferences.
    Preserved exactly from original for test compatibility.
    """
    score = 0.0
    reasons = []

    if song['genre'] == user_prefs['favorite_genre'].lower():
        score += 1.0
        reasons.append("genre match (+1.0)")

    if song['mood'] == user_prefs['favorite_mood'].lower():
        score += 1.0
        reasons.append("mood match (+1.0)")

    energy_gap = abs(song['energy'] - user_prefs['target_energy'])
    energy_points = max(0.0, 1.0 - energy_gap) * 2.0
    score += energy_points
    reasons.append(f"energy match (+{energy_points:.2f})")

    return round(score, 2), reasons


def recommend_songs(user_prefs: Dict, songs, k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Dict-based recommender. Accepts both Song objects and dicts.
    Preserved for backward compatibility with run_simulation.py.
    """
    # Normalize to dicts if Song objects were passed
    if songs and isinstance(songs[0], Song):
        songs = songs_to_dicts(songs)

    scored_catalog = []
    for song in songs:
        final_score, reason_list = score_song(user_prefs, song)
        formatted_reasons = ", ".join(reason_list)
        scored_catalog.append((song, final_score, formatted_reasons))

    ranked_songs = sorted(scored_catalog, key=lambda x: x[1], reverse=True)
    return ranked_songs[:k]
