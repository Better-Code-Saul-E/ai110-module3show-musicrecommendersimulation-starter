# 🎵 Applied AI Music Recommender System

> **Module 5 Final Project** — Extended from the Module 1–3 Music Recommender prototype into a full applied AI system with RAG retrieval, agentic planning, reliability testing, and production-grade guardrails.

---

## Original Project

This system evolves the **Module 1–3 Music Recommender**, which scored songs against user preference profiles (genre, mood, target energy) using a weighted scoring algorithm. The original system could recommend songs from a small hand-crafted CSV but lacked real data, natural language understanding, and reliability infrastructure.

---

## What's New in This Version

| Feature | Status |
|---|---|
| Real-data pipeline (Kaggle or generated) | ✅ `src/data_loader.py` |
| RAG retrieval (natural language queries) | ✅ `src/rag_retriever.py` |
| Agentic workflow with observable steps | ✅ `src/agent.py` |
| Input guardrails + structured logging | ✅ `src/guardrails.py` |
| Reliability test harness (13 test cases) | ✅ `src/evaluator.py` |
| 30 unit tests | ✅ `tests/test_recommender.py` |

---

## System Architecture

```
User Input (profile or natural language)
        │
        ▼
┌───────────────────┐
│  guardrails.py    │  ← Input validation, sanitization, rate limiting
└────────┬──────────┘
         │ validated input
         ▼
┌───────────────────┐      ┌─────────────────────┐
│   agent.py        │◄─────│  rag_retriever.py   │
│  (Planner)        │      │  TF-IDF index        │
│  7-step chain:    │      │  Query expansion     │
│  1. parse_intent  │      │  Semantic retrieval  │
│  2. rag_retrieve  │      └─────────────────────┘
│  3. score_rank    │
│  4. check_diversity│     ┌─────────────────────┐
│  5. confidence    │◄─────│  recommender.py     │
│  6. retry?        │      │  Weighted scoring   │
│  7. explain       │      │  Confidence scoring  │
└────────┬──────────┘      └─────────────────────┘
         │
         ▼
┌───────────────────┐
│  data_loader.py   │  ← Kaggle CSV or generated realistic dataset
└───────────────────┘
         │
         ▼                 ┌─────────────────────┐
   AgentResult             │   evaluator.py      │
   (songs +                │   13 test cases     │
    confidence +           │   pass/fail scores  │
    reasoning trace)       │   JSON export       │
                           └─────────────────────┘
```

**Data flows:** raw query → guardrails → agent planner → RAG retriever (candidates) → scorer (ranked) → diversity/confidence check → [retry if needed] → explanations → output

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/applied-ai-music-recommender.git
cd applied-ai-music-recommender

# 2. No external dependencies required!
#    All modules use Python standard library only.
#    (Optional: pip install kaggle  — only if using real Kaggle data)

# 3. Run the full demo
python run_simulation.py --mode both

# 4. Run with a custom query
python run_simulation.py --query "chill jazz for a rainy afternoon"

# 5. Run the test harness
python -m src.evaluator

# 6. Run unit tests
python -m unittest tests/test_recommender.py -v
```

### Using Real Kaggle Data (Optional)

The system works out-of-the-box with a statistically realistic generated dataset. To use the real 114k-song Kaggle dataset:

```bash
pip install kaggle
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
unzip spotify-tracks-dataset.zip
mv dataset.csv data/spotify_tracks.csv
python run_simulation.py  # auto-detects and loads real data
```

---

## Sample Interactions

### Classic Profile Mode
```
Profile: High-Energy Pop | Genre: pop | Mood: happy | Energy: 0.9
  ♪  Fire Dream [Radio Edit] — Ed Sheeran
     Score: 4.00 | genre match (+1.0), mood match (+1.0), energy match (+2.00)
  ♪  Night Gold — The Weeknd
     Score: 3.97 | genre match, mood match, energy match (+1.97)
```

### Agentic Mode (Natural Language)
```
Query: "chill lo-fi beats for a late night study session"

  📋 Intent: genre=lo-fi, mood=chill, energy=0.3
  🔍 RAG: 40 candidates retrieved (confidence=0.44)
  🎵 Top pick: 'Drift Haze Pt. II' by Øneheart
  📊 Confidence: [████████░░] 0.87

  1. Drift Haze Pt. II — Øneheart  [lo-fi / chill / energy 0.35]
     Why: matches your lo-fi preference, chill mood, energy level spot-on, acoustic texture
  2. Quiet Study — Idealism  [lo-fi / chill / energy 0.36]
  3. Warm Quiet — Idealism   [lo-fi / chill / energy 0.38]
```

### Test Harness
```
  ✅ [happy_pop_returns_results]       (1ms)
  ✅ [lofi_chill_genre_match]          (1ms)
  ✅ [rock_intense_energy]             (0ms)
  ✅ [natural_query_study_session]     (7ms)
  ✅ [adversarial_conflicting_prefs]   (1ms)
  ✅ [adversarial_invalid_energy]      (1ms)
  ...
  Pass rate: 100% | Avg confidence: 0.697
```

---

## Design Decisions

**Why TF-IDF instead of neural embeddings for RAG?**
Neural embeddings (OpenAI, Cohere) would require an API key and network call per query — breaking offline use and adding latency. TF-IDF runs in ~1ms, is fully deterministic, and is interpretable: you can see exactly which terms drove each match. The interface is abstraction-compatible: swapping in embeddings later requires only changing `TFIDFIndex.query()`.

**Why mood derivation instead of a mood column?**
Real Spotify datasets don't include mood labels. Deriving mood from energy + valence mirrors the Russell Circumplex Model of Affect — a research-backed framework for mapping audio features to emotional states. This makes the system generalize to any Spotify-format dataset.

**Why a 7-step agent instead of a simple function call?**
Each step is observable, logged, and independently testable. The retry mechanism (Step 6) provides real adaptive behavior: if confidence drops below 0.45, the agent broadens its genre constraint and re-searches. This is meaningfully different from a lookup function.

**Trade-offs made:**
- Generated data uses real artist names and genre-accurate audio feature distributions, but songs are procedurally titled. Loading real Kaggle data replaces this entirely.
- The diversity check flags but doesn't force-rerank results — preserving scoring integrity over aesthetic variety.
- Rate limiting is per-session only; a deployed version would need Redis or a DB-backed store.

---

## Testing Summary

**Unit tests (30 total, 30 passed):**
- `TestScoreSong` — scoring algorithm correctness
- `TestRecommender` — OOP API, confidence scoring
- `TestRecommendSongs` — dict-based backward-compat API
- `TestDataLoader` — generation, field validation, mood derivation
- `TestRAGRetriever` — index building, query, expansion, error handling
- `TestGuardrails` — validation, injection blocking, clamping, defaults

**Reliability harness (13 cases, 13 passed):**
- Functional: genre/mood/energy correctness for 5 profiles
- RAG/NL: 3 natural language query tests
- Adversarial: conflicting prefs, empty query, invalid energy
- Consistency: determinism across runs
- Diversity: multi-artist output

**What didn't work initially:**
- The `validate_user_prefs()` function crashed on non-dict input because `dict()` was called before the type check — caught by the test harness.
- The diversity check flagged all lo-fi queries as "low diversity" because the genre-matched catalog is intentionally narrow. This is correct behavior, not a bug; the warning is surfaced to the user without changing results.

---

## Reflection & Ethics

**Limitations:**
- Mood labels are derived, not ground-truth. A song labeled "intense" by energy/valence heuristics might subjectively feel "melancholic" to some listeners.
- The generated dataset uses real artist names but synthetic song titles. This means artist-name queries ("give me something like Drake") work, but title-based searches won't match real songs.
- TF-IDF has no semantic understanding — "happy" and "joyful" are related only if both appear in the same document, not by meaning.

**Misuse potential:**
- A recommendation system could be manipulated to only surface certain artists (payola). Mitigation: scoring is fully transparent and logged; genre/mood weights are published.
- Natural language parsing could be prompted with injection attempts. Mitigation: `validate_query()` blocks known injection patterns before any processing.

**Surprises during testing:**
- The adversarial "ambient + intense + 0.95 energy" profile still returned results because energy proximity is the dominant signal when genre/mood both miss. The system degrades gracefully rather than returning nothing.
- Confidence scores were higher than expected (avg 0.697) — the TF-IDF retrieval pre-filters candidates well enough that the scorer finds strong matches even for unusual queries.

**AI collaboration:**
- *Helpful:* Claude suggested deriving mood from the Russell Circumplex Model, which turned out to be a cleaner and more defensible approach than manual mood mapping.
- *Incorrect:* An early suggestion used `sklearn.TfidfVectorizer` as the RAG backend, which would have added a dependency. The from-scratch TF-IDF implementation is more appropriate for a zero-dependency educational project.
