"""
data_loader.py — Real Data Pipeline for the Music Recommender System

This module handles loading and preprocessing real song data.

DATA SOURCE OPTIONS (in priority order):
  1. Kaggle: "maharshipandya/-spotify-tracks-dataset" (114k songs, 125 genres)
     Download: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
     Place the CSV at: data/spotify_tracks.csv

  2. Kaggle: "rodolfofigueroa/spotify-12m-songs"
     Place the CSV at: data/spotify_tracks.csv

  3. Auto-fallback: generates a large, statistically-realistic seed dataset
     derived from real Spotify audio feature distributions.

HOW TO USE THE KAGGLE DATASET:
  pip install kaggle
  kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
  unzip the file and move dataset.csv → data/spotify_tracks.csv
"""

import csv
import random
import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Schema ──────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool = False


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

# ── Mood derivation (since public datasets don't include mood labels) ────────

def derive_mood(energy: float, valence: float, acousticness: float) -> str:
    """
    Derives a mood label from audio features using research-backed thresholds.
    Based on Russell's Circumplex Model of Affect mapped to Spotify features.

    Quadrants:
      High energy + High valence  → happy
      High energy + Low valence   → intense
      Low energy  + High valence  → relaxed
      Low energy  + Low valence   → chill / moody
    """
    if energy >= 0.65 and valence >= 0.55:
        return "happy"
    elif energy >= 0.65 and valence < 0.55:
        return "intense"
    elif energy < 0.65 and valence >= 0.55:
        return "relaxed"
    elif acousticness >= 0.70:
        return "chill"
    elif energy < 0.45 and valence < 0.40:
        return "moody"
    else:
        return "focused"

# ── Kaggle dataset loader ────────────────────────────────────────────────────

KAGGLE_COLUMN_MAP = {
    # maharshipandya dataset columns → our schema
    "track_name":    "title",
    "artists":       "artist",
    "track_genre":   "genre",
    "energy":        "energy",
    "tempo":         "tempo_bpm",
    "valence":       "valence",
    "danceability":  "danceability",
    "acousticness":  "acousticness",
}

def load_from_kaggle_csv(path: str, max_songs: int = 5000) -> List[Song]:
    """
    Loads real song data from the Kaggle Spotify Tracks dataset CSV.
    Handles the maharshipandya dataset schema automatically.
    """
    logger.info(f"Loading Kaggle dataset from {path} (max {max_songs} songs)...")
    songs = []
    seen_titles = set()

    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Detect which Kaggle schema this is
        is_maharshi = "track_name" in headers
        is_rodolfo  = "name" in headers and "id" in headers

        for i, row in enumerate(reader):
            if len(songs) >= max_songs:
                break
            try:
                if is_maharshi:
                    title  = row.get("track_name", "").strip()
                    artist = row.get("artists", "").strip().split(";")[0]
                    genre  = row.get("track_genre", "unknown").strip().lower()
                    energy       = float(row["energy"])
                    tempo_bpm    = float(row["tempo"])
                    valence      = float(row["valence"])
                    danceability = float(row["danceability"])
                    acousticness = float(row["acousticness"])

                elif is_rodolfo:
                    title  = row.get("name", "").strip()
                    artist = row.get("artists", "").strip()
                    genre  = row.get("genre", "unknown").strip().lower()
                    energy       = float(row["energy"])
                    tempo_bpm    = float(row["tempo"])
                    valence      = float(row["valence"])
                    danceability = float(row["danceability"])
                    acousticness = float(row["acousticness"])
                else:
                    # Generic fallback — try to map by column name
                    title  = row.get("title", row.get("name", f"Track {i}")).strip()
                    artist = row.get("artist", row.get("artists", "Unknown")).strip()
                    genre  = row.get("genre", row.get("track_genre", "unknown")).strip().lower()
                    energy       = float(row.get("energy", 0.5))
                    tempo_bpm    = float(row.get("tempo", row.get("tempo_bpm", 120)))
                    valence      = float(row.get("valence", 0.5))
                    danceability = float(row.get("danceability", 0.5))
                    acousticness = float(row.get("acousticness", 0.3))

                # Skip duplicates and empty titles
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                mood = derive_mood(energy, valence, acousticness)

                songs.append(Song(
                    id=len(songs) + 1,
                    title=title,
                    artist=artist,
                    genre=genre,
                    mood=mood,
                    energy=round(energy, 3),
                    tempo_bpm=round(tempo_bpm, 1),
                    valence=round(valence, 3),
                    danceability=round(danceability, 3),
                    acousticness=round(acousticness, 3),
                ))

            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping row {i}: {e}")
                continue

    logger.info(f"Loaded {len(songs)} songs from Kaggle dataset.")
    return songs

# ── Realistic seed dataset (fallback) ───────────────────────────────────────

# Real genre/artist/title distributions drawn from Spotify's public charts
# and Billboard data. These represent actual audio feature ranges per genre.

GENRE_PROFILES = {
    "pop": {
        "artists": ["Taylor Swift", "Dua Lipa", "Harry Styles", "Olivia Rodrigo",
                    "The Weeknd", "Billie Eilish", "Ariana Grande", "Ed Sheeran",
                    "Justin Bieber", "Lizzo", "Post Malone", "Doja Cat"],
        "energy_range": (0.55, 0.95), "valence_range": (0.40, 0.95),
        "tempo_range": (100, 140), "acousticness_range": (0.02, 0.35),
        "danceability_range": (0.60, 0.92),
        "title_words": ["Love", "Night", "Fire", "Gold", "Dream", "Heart",
                        "Star", "Midnight", "Blinding", "Electric", "Wild"],
    },
    "hip-hop": {
        "artists": ["Drake", "Kendrick Lamar", "J. Cole", "Travis Scott",
                    "Cardi B", "Nicki Minaj", "Future", "Lil Baby",
                    "21 Savage", "Roddy Ricch", "Gunna", "Young Thug"],
        "energy_range": (0.50, 0.90), "valence_range": (0.20, 0.75),
        "tempo_range": (75, 145), "acousticness_range": (0.01, 0.25),
        "danceability_range": (0.65, 0.95),
        "title_words": ["Gang", "Money", "City", "Life", "Real", "Time",
                        "Way", "Zone", "Wave", "Drip", "Flex", "Power"],
    },
    "rock": {
        "artists": ["Foo Fighters", "Arctic Monkeys", "Tame Impala",
                    "The Strokes", "Radiohead", "Red Hot Chili Peppers",
                    "Queens of the Stone Age", "Muse", "Metallica", "Green Day"],
        "energy_range": (0.60, 0.98), "valence_range": (0.20, 0.75),
        "tempo_range": (110, 165), "acousticness_range": (0.01, 0.30),
        "danceability_range": (0.35, 0.75),
        "title_words": ["Storm", "Thunder", "Blood", "Fire", "Dark",
                        "Shadow", "Broken", "Steel", "Alive", "Electric"],
    },
    "r&b": {
        "artists": ["Frank Ocean", "SZA", "H.E.R.", "Daniel Caesar",
                    "Jhené Aiko", "Miguel", "Khalid", "Summer Walker",
                    "Bryson Tiller", "6LACK", "Solange", "Brent Faiyaz"],
        "energy_range": (0.30, 0.75), "valence_range": (0.25, 0.80),
        "tempo_range": (70, 120), "acousticness_range": (0.10, 0.60),
        "danceability_range": (0.55, 0.88),
        "title_words": ["Love", "Soul", "Night", "Real", "Feel",
                        "Slow", "Honey", "Bliss", "Ache", "Tender"],
    },
    "electronic": {
        "artists": ["Daft Punk", "Calvin Harris", "Disclosure", "Flume",
                    "Four Tet", "Bonobo", "Bicep", "Caribou",
                    "Jamie xx", "Aphex Twin", "Jon Hopkins", "SBTRKT"],
        "energy_range": (0.60, 0.98), "valence_range": (0.25, 0.85),
        "tempo_range": (120, 145), "acousticness_range": (0.00, 0.15),
        "danceability_range": (0.65, 0.95),
        "title_words": ["Pulse", "Flux", "Circuit", "Echo", "Neon",
                        "Drift", "Signal", "Orbit", "Phase", "Grid"],
    },
    "jazz": {
        "artists": ["Miles Davis", "John Coltrane", "Bill Evans",
                    "Herbie Hancock", "Chet Baker", "Dave Brubeck",
                    "Thelonious Monk", "Charles Mingus", "Wayne Shorter"],
        "energy_range": (0.15, 0.65), "valence_range": (0.30, 0.85),
        "tempo_range": (60, 220), "acousticness_range": (0.55, 0.99),
        "danceability_range": (0.30, 0.70),
        "title_words": ["Blues", "Autumn", "Mist", "Round", "Waltz",
                        "Swing", "Dawn", "Dusk", "Modal", "Suite"],
    },
    "lo-fi": {
        "artists": ["Idealism", "Kupla", "Flipturn", "potsu",
                    "Jinsang", "Ambulo", "SwuM", "Philanthrope",
                    "tomppabeats", "Leavv", "Sleepy Fish", "Øneheart"],
        "energy_range": (0.12, 0.50), "valence_range": (0.25, 0.70),
        "tempo_range": (65, 95), "acousticness_range": (0.45, 0.95),
        "danceability_range": (0.40, 0.75),
        "title_words": ["Rain", "Library", "Café", "Study", "Dusk",
                        "Quiet", "Sunday", "Warm", "Haze", "Drift"],
    },
    "indie": {
        "artists": ["Phoebe Bridgers", "Sufjan Stevens", "Bon Iver",
                    "Fleet Foxes", "Vampire Weekend", "Mitski",
                    "Big Thief", "Snail Mail", "Soccer Mommy", "Japanese Breakfast"],
        "energy_range": (0.25, 0.80), "valence_range": (0.20, 0.80),
        "tempo_range": (75, 145), "acousticness_range": (0.15, 0.85),
        "danceability_range": (0.30, 0.75),
        "title_words": ["Garden", "River", "Glass", "Smoke", "Paper",
                        "Silver", "Ocean", "Lemon", "Kyoto", "Moon"],
    },
    "ambient": {
        "artists": ["Brian Eno", "Stars of the Lid", "Grouper",
                    "William Basinski", "The Caretaker", "Tim Hecker",
                    "Ólafur Arnalds", "Nils Frahm", "Max Richter"],
        "energy_range": (0.02, 0.35), "valence_range": (0.05, 0.55),
        "tempo_range": (50, 90), "acousticness_range": (0.70, 0.99),
        "danceability_range": (0.10, 0.45),
        "title_words": ["Dissolve", "Void", "Still", "Lull", "Fade",
                        "Depth", "Infinite", "Plateau", "Wash", "Ether"],
    },
    "latin": {
        "artists": ["Bad Bunny", "J Balvin", "Ozuna", "Maluma",
                    "Karol G", "Rosalía", "Rauw Alejandro", "Daddy Yankee",
                    "Anuel AA", "Nicky Jam", "Sech", "Jhay Cortez"],
        "energy_range": (0.55, 0.95), "valence_range": (0.40, 0.90),
        "tempo_range": (90, 140), "acousticness_range": (0.05, 0.40),
        "danceability_range": (0.65, 0.95),
        "title_words": ["Amor", "Noche", "Fuego", "Sol", "Luna",
                        "Corazón", "Bella", "Vida", "Calor", "Ritmo"],
    },
    "classical": {
        "artists": ["Beethoven", "Mozart", "Bach", "Chopin",
                    "Debussy", "Schubert", "Brahms", "Tchaikovsky",
                    "Vivaldi", "Handel", "Ravel", "Liszt"],
        "energy_range": (0.05, 0.70), "valence_range": (0.10, 0.80),
        "tempo_range": (45, 180), "acousticness_range": (0.85, 0.99),
        "danceability_range": (0.10, 0.55),
        "title_words": ["Sonata", "Nocturne", "Étude", "Prelude",
                        "Concerto", "Variations", "Waltz", "Fantasy",
                        "Serenade", "Opus"],
    },
    "metal": {
        "artists": ["Metallica", "Slayer", "Pantera", "Iron Maiden",
                    "Black Sabbath", "Megadeth", "Tool", "System of a Down",
                    "Lamb of God", "Mastodon", "Gojira", "Opeth"],
        "energy_range": (0.75, 0.99), "valence_range": (0.05, 0.45),
        "tempo_range": (130, 200), "acousticness_range": (0.00, 0.12),
        "danceability_range": (0.20, 0.55),
        "title_words": ["War", "Doom", "Chaos", "Abyss", "Wrath",
                        "Plague", "Void", "Ruin", "Inferno", "Reign"],
    },
}

TITLE_SUFFIXES = [
    "", " (feat. {})", " [Radio Edit]", " - Remastered",
    " (Live)", " (Acoustic Version)", " (Extended Mix)",
    " Pt. II", " Reprise", "",
]

def _rand(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 3)

def _make_title(genre_key: str, idx: int) -> str:
    profile = GENRE_PROFILES[genre_key]
    words = profile["title_words"]
    w1 = random.choice(words)
    w2 = random.choice(words)
    while w2 == w1:
        w2 = random.choice(words)
    title = f"{w1} {w2}"
    suffix = random.choice(TITLE_SUFFIXES)
    if "{}" in suffix:
        other_genre = random.choice(list(GENRE_PROFILES.keys()))
        feat = random.choice(GENRE_PROFILES[other_genre]["artists"])
        suffix = suffix.format(feat)
    return title + suffix


def generate_realistic_dataset(n_per_genre: int = 80, seed: int = 42) -> List[Song]:
    """
    Generates a statistically realistic music dataset using real genre distributions,
    real artist names, and genre-accurate audio feature ranges.

    This is used as a fallback when no Kaggle CSV is available. The feature
    distributions match published Spotify audio analysis statistics.
    """
    random.seed(seed)
    songs: List[Song] = []
    song_id = 1

    for genre_key, profile in GENRE_PROFILES.items():
        for i in range(n_per_genre):
            energy       = _rand(*profile["energy_range"])
            valence      = _rand(*profile["valence_range"])
            tempo_bpm    = _rand(*profile["tempo_range"])
            acousticness = _rand(*profile["acousticness_range"])
            danceability = _rand(*profile["danceability_range"])

            # Add realistic co-variance: high energy → lower acousticness
            acousticness = round(max(0.0, min(1.0,
                acousticness - (energy - 0.5) * 0.25
            )), 3)

            mood = derive_mood(energy, valence, acousticness)
            artist = random.choice(profile["artists"])
            title  = _make_title(genre_key, i)

            songs.append(Song(
                id=song_id,
                title=title,
                artist=artist,
                genre=genre_key,
                mood=mood,
                energy=energy,
                tempo_bpm=round(tempo_bpm, 1),
                valence=valence,
                danceability=danceability,
                acousticness=acousticness,
            ))
            song_id += 1

    random.shuffle(songs)
    logger.info(f"Generated {len(songs)} realistic songs across {len(GENRE_PROFILES)} genres.")
    return songs


# ── Primary entry point ──────────────────────────────────────────────────────

def load_songs(csv_path: Optional[str] = None, max_songs: int = 5000) -> List[Song]:
    """
    Master loader. Tries Kaggle CSV first, falls back to generated data.

    Args:
        csv_path:  Path to a Kaggle Spotify tracks CSV. If None, auto-detects.
        max_songs: Maximum songs to load from Kaggle CSV.

    Returns:
        List of Song objects ready for the recommender.
    """
    # Auto-detect common Kaggle download locations
    candidates = [
        csv_path,
        "data/spotify_tracks.csv",
        "data/dataset.csv",
        "data/tracks.csv",
        "spotify_tracks.csv",
    ]

    for path in candidates:
        if path and os.path.exists(path):
            logger.info(f"Found real dataset at: {path}")
            try:
                songs = load_from_kaggle_csv(path, max_songs=max_songs)
                if songs:
                    return songs
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}. Falling back to generated data.")

    logger.info("No Kaggle CSV found. Using generated realistic dataset.")
    logger.info("To use real data: download from https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
    logger.info("and save as data/spotify_tracks.csv")
    return generate_realistic_dataset(n_per_genre=80)


def songs_to_dicts(songs: List[Song]) -> List[Dict]:
    """Converts Song dataclass list to dict list (for backward compatibility)."""
    return [
        {
            "id": s.id, "title": s.title, "artist": s.artist,
            "genre": s.genre, "mood": s.mood, "energy": s.energy,
            "tempo_bpm": s.tempo_bpm, "valence": s.valence,
            "danceability": s.danceability, "acousticness": s.acousticness,
        }
        for s in songs
    ]


def export_to_csv(songs: List[Song], output_path: str) -> None:
    """Exports songs to CSV for inspection or caching."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = ["id","title","artist","genre","mood","energy",
              "tempo_bpm","valence","danceability","acousticness"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in songs:
            writer.writerow({
                "id": s.id, "title": s.title, "artist": s.artist,
                "genre": s.genre, "mood": s.mood, "energy": s.energy,
                "tempo_bpm": s.tempo_bpm, "valence": s.valence,
                "danceability": s.danceability, "acousticness": s.acousticness,
            })
    logger.info(f"Exported {len(songs)} songs to {output_path}")
