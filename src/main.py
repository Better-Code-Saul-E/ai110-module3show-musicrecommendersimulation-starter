"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    test_profiles = {
        "High-Energy Pop": {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.90},
        "Chill Lofi": {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.30},
        "Deep Intense Rock": {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.85},
        "The Conflicted User (Adversarial)": {"favorite_genre": "ambient", "favorite_mood": "intense", "target_energy": 0.95} 
    }

    for profile_name, user_prefs in test_profiles.items():
        print(f"\n{'='*40}")
        print(f"Testing Profile: {profile_name}")
        print(f"{'='*40}")
        
        recommendations = recommend_songs(user_prefs, songs, k=3) # Showing top 3 for brevity

        for rec in recommendations:
            song, score, explanation = rec
            print(f"{song['title']} by {song['artist']} - Score: {score:.2f}")
            print(f"   Because: {explanation}\n")

if __name__ == "__main__":
    main()
