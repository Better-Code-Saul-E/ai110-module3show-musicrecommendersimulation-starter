import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file and converts numeric strings to floats/ints.
    """
    print(f"Loading songs from {csv_path}...")
    songs = []
    
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert categorical strings to lowercase for safer matching
            row['genre'] = row['genre'].lower()
            row['mood'] = row['mood'].lower()
            
            # Convert numerical strings to proper data types
            row['id'] = int(row['id'])
            row['energy'] = float(row['energy'])
            row['tempo_bpm'] = float(row['tempo_bpm'])
            row['valence'] = float(row['valence'])
            row['danceability'] = float(row['danceability'])
            row['acousticness'] = float(row['acousticness'])
            
            songs.append(row)
            
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences based on the Algorithm Recipe.
    Returns the numeric score and a list of reasons.
    """
    score = 0.0
    reasons = []

    # 1. Genre Match (+2.0 points)
    if song['genre'] == user_prefs['favorite_genre'].lower():
        score += 2.0
        reasons.append("genre match (+2.0)")

    # 2. Mood Match (+1.0 point)
    if song['mood'] == user_prefs['favorite_mood'].lower():
        score += 1.0
        reasons.append("mood match (+1.0)")

    # 3. Energy Gap (+ up to 1.0 point)
    # The closer the song's energy is to the target, the smaller the gap.
    energy_gap = abs(song['energy'] - user_prefs['target_energy'])
    energy_points = max(0.0, 1.0 - energy_gap) # Ensure we don't drop below 0 points
    score += energy_points
    reasons.append(f"energy match (+{energy_points:.2f})")

    return round(score, 2), reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Scores all songs, ranks them descending by score, and returns the top k.
    """
    scored_catalog = []
    
    # Evaluate every song using our scoring judge
    for song in songs:
        final_score, reason_list = score_song(user_prefs, song)
        formatted_reasons = ", ".join(reason_list)
        scored_catalog.append((song, final_score, formatted_reasons))
    
    # Sort the catalog by the score (index 1 of the tuple) in descending order
    ranked_songs = sorted(scored_catalog, key=lambda x: x[1], reverse=True)
    
    return ranked_songs[:k]