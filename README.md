# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.

1) Each song will be treated as categorical or numical based on its values like a scale from 0.0-1.0 or  'pop', 'rock'

2) The profile stores the users preferences based on the songs attributes they tend to like. A favorite genre, favorite mood, and target energy.

3) The recommender will calculate a numerical score for a single song based ont the users preferences. For Categorical properties, it will give +2.0 points for a matching genre, and +1.0 point for a matching mood. For Numberical properties it will calculate the difference between the songs energy and the users target energy. If there is a smaller gap more points will be rewarded.

4) It will score every song in the catalog, then it will order them be descending order. The top results will be selected and presented.

graph TD
    A[User Profile] --> C{Scoring Loop: Evaluate Every Song}
    B[songs.csv Catalog] --> C
    
    C --> D{Check Genre}
    D -->|Match| D1[+2.0 Points]
    D -->|No Match| D2[+0 Points]
    
    C --> E{Check Mood}
    E -->|Match| E1[+1.0 Point]
    E -->|No Match| E2[+0 Points]
    
    C --> F{Calculate Energy Gap}
    F --> F1[+ 1.0 - abs(song_energy - target_energy)]
    
    D1 --> G
    D2 --> G
    E1 --> G
    E2 --> G
    F1 --> G
    
    G[Sum Total Score for Song] --> H[Ranking Rule: Sort All Songs Descending]
    H --> I[Output: Top 5 Recommendations]

The Final algorithm recipe
The system calculates a score for each track using:
Genre match +2.0 points
Mood match +1.0 points
Energy proximity up to +1.0 points

Some of the potential biases:
Genre match: The genre is weighted twice as heavy as mood or energy, The system might ignore a good match because it determines the user prefers a genre
Missing context: The system does not account for more favorite generes and it doesnt adapt to the recent history


---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

