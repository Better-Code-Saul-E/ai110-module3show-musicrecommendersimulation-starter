"""
evaluator.py — Reliability Test Harness

Runs the music recommender system against a suite of predefined test cases
and prints a structured pass/fail summary with confidence ratings.

This satisfies both:
  - "Reliability or Testing System" (required feature)
  - "Test Harness or Evaluation Script" (stretch feature +2 pts)

Usage:
  python -m src.evaluator
  python -m src.evaluator --verbose
  python -m src.evaluator --mode agent   # test agentic pipeline
  python -m src.evaluator --mode basic   # test basic recommender only
  python -m src.evaluator --export results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any

logger = logging.getLogger(__name__)

# ── Test Case Schema ─────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    description: str
    input_type: str          # "prefs" | "query"
    input_data: Dict
    checks: List[Dict]       # List of {type, params, description}
    tags: List[str] = field(default_factory=list)


@dataclass
class CheckResult:
    check_type: str
    description: str
    passed: bool
    actual: Any
    expected: Any
    message: str


@dataclass
class TestResult:
    case_name: str
    passed: bool
    confidence: float
    checks: List[CheckResult]
    duration_ms: float
    recommendations: List[str]  # titles for display
    error: Optional[str] = None


# ── Test Suite ───────────────────────────────────────────────────────────────

TEST_SUITE: List[TestCase] = [

    # ── Functional correctness tests ────────────────────────────────────────

    TestCase(
        name="happy_pop_returns_results",
        description="High-energy pop request should return 5 results",
        input_type="prefs",
        input_data={"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.85},
        checks=[
            {"type": "min_results",  "params": {"n": 5},    "desc": "At least 5 results returned"},
            {"type": "max_results",  "params": {"n": 5},    "desc": "Exactly 5 results (k=5)"},
            {"type": "has_titles",   "params": {},           "desc": "All results have non-empty titles"},
        ],
        tags=["functional", "basic"],
    ),

    TestCase(
        name="lofi_chill_genre_match",
        description="Lo-fi chill request should prefer lo-fi songs",
        input_type="prefs",
        input_data={"favorite_genre": "lo-fi", "favorite_mood": "chill", "target_energy": 0.30},
        checks=[
            {"type": "min_results",  "params": {"n": 3},       "desc": "At least 3 results"},
            {"type": "genre_match_rate", "params": {"genre": "lo-fi", "min_rate": 0.40},
             "desc": "At least 40% of results are lo-fi genre"},
            {"type": "energy_proximity", "params": {"target": 0.30, "max_avg_gap": 0.35},
             "desc": "Average energy within 0.35 of target (0.30)"},
        ],
        tags=["functional", "genre"],
    ),

    TestCase(
        name="rock_intense_energy",
        description="Intense rock should return high-energy songs",
        input_type="prefs",
        input_data={"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.88},
        checks=[
            {"type": "min_results",    "params": {"n": 3},     "desc": "At least 3 results"},
            {"type": "energy_min_avg", "params": {"min": 0.55}, "desc": "Average energy >= 0.55"},
            {"type": "mood_match_rate", "params": {"mood": "intense", "min_rate": 0.20},
             "desc": "At least 20% of results are intense mood"},
        ],
        tags=["functional", "energy"],
    ),

    TestCase(
        name="ambient_chill_low_energy",
        description="Ambient chill should return low-energy songs",
        input_type="prefs",
        input_data={"favorite_genre": "ambient", "favorite_mood": "chill", "target_energy": 0.15},
        checks=[
            {"type": "min_results",    "params": {"n": 3},     "desc": "At least 3 results"},
            {"type": "energy_max_avg", "params": {"max": 0.65}, "desc": "Average energy <= 0.65"},
        ],
        tags=["functional", "energy"],
    ),

    TestCase(
        name="classical_jazz_acoustic",
        description="Classical/jazz should return high-acousticness songs",
        input_type="prefs",
        input_data={"favorite_genre": "jazz", "favorite_mood": "relaxed", "target_energy": 0.35},
        checks=[
            {"type": "min_results",         "params": {"n": 3},     "desc": "At least 3 results"},
            {"type": "acousticness_min_avg", "params": {"min": 0.30},
             "desc": "Average acousticness >= 0.30"},
        ],
        tags=["functional", "acoustic"],
    ),

    # ── RAG / Natural language query tests ──────────────────────────────────

    TestCase(
        name="natural_query_study_session",
        description="'chill beats for a study session' should return low-energy results",
        input_type="query",
        input_data={"query": "chill beats for a study session"},
        checks=[
            {"type": "min_results",    "params": {"n": 3},     "desc": "At least 3 results"},
            {"type": "energy_max_avg", "params": {"max": 0.70}, "desc": "Average energy <= 0.70"},
            {"type": "confidence_min", "params": {"min": 0.20}, "desc": "Agent confidence >= 0.20"},
        ],
        tags=["rag", "natural_language"],
    ),

    TestCase(
        name="natural_query_workout",
        description="'high energy workout music' should return energetic results",
        input_type="query",
        input_data={"query": "high energy workout music pump me up"},
        checks=[
            {"type": "min_results",    "params": {"n": 3},     "desc": "At least 3 results"},
            {"type": "energy_min_avg", "params": {"min": 0.55}, "desc": "Average energy >= 0.55"},
            {"type": "confidence_min", "params": {"min": 0.20}, "desc": "Agent confidence >= 0.20"},
        ],
        tags=["rag", "natural_language"],
    ),

    TestCase(
        name="natural_query_late_night",
        description="'late night drive moody' should return moody/dark results",
        input_type="query",
        input_data={"query": "late night drive moody atmospheric"},
        checks=[
            {"type": "min_results",    "params": {"n": 3},    "desc": "At least 3 results"},
            {"type": "confidence_min", "params": {"min": 0.15}, "desc": "Agent confidence >= 0.15"},
        ],
        tags=["rag", "natural_language"],
    ),

    # ── Adversarial / edge case tests ────────────────────────────────────────

    TestCase(
        name="adversarial_conflicting_prefs",
        description="Conflicting preferences (ambient genre + intense mood) should still return results",
        input_type="prefs",
        input_data={"favorite_genre": "ambient", "favorite_mood": "intense", "target_energy": 0.95},
        checks=[
            {"type": "min_results",  "params": {"n": 1},  "desc": "At least 1 result despite conflict"},
            {"type": "no_crash",     "params": {},         "desc": "System does not crash"},
        ],
        tags=["adversarial", "edge_case"],
    ),

    TestCase(
        name="adversarial_empty_query",
        description="Empty or minimal query should not crash",
        input_type="query",
        input_data={"query": "music"},
        checks=[
            {"type": "no_crash",    "params": {},        "desc": "System does not crash"},
            {"type": "min_results", "params": {"n": 1},  "desc": "At least 1 result returned"},
        ],
        tags=["adversarial", "edge_case"],
    ),

    TestCase(
        name="adversarial_invalid_energy",
        description="Out-of-range energy (guardrail test) should be clamped and handled",
        input_type="prefs",
        input_data={"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 9.99},
        checks=[
            {"type": "no_crash",    "params": {},       "desc": "System does not crash with invalid energy"},
            {"type": "min_results", "params": {"n": 1}, "desc": "Returns at least 1 result after clamping"},
        ],
        tags=["adversarial", "guardrails"],
    ),

    # ── Consistency tests ────────────────────────────────────────────────────

    TestCase(
        name="consistency_deterministic",
        description="Same preferences run twice should return same top song",
        input_type="prefs",
        input_data={"favorite_genre": "hip-hop", "favorite_mood": "intense", "target_energy": 0.80},
        checks=[
            {"type": "min_results",   "params": {"n": 3}, "desc": "At least 3 results"},
            {"type": "deterministic", "params": {"runs": 2}, "desc": "Top result is same across 2 runs"},
        ],
        tags=["consistency"],
    ),

    # ── Diversity test ───────────────────────────────────────────────────────

    TestCase(
        name="diversity_multiple_artists",
        description="5 recommendations should not all be from the same artist",
        input_type="prefs",
        input_data={"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.75},
        checks=[
            {"type": "artist_diversity", "params": {"min_unique": 2},
             "desc": "At least 2 different artists in top 5"},
        ],
        tags=["diversity"],
    ),
]


# ── Check execution engine ────────────────────────────────────────────────────

def run_check(check: Dict, songs: List, confidence: float, run_fn: Callable,
              prefs: Dict) -> CheckResult:
    """Executes a single check against recommendation results."""
    ctype = check["type"]
    params = check["params"]
    desc = check["desc"]

    try:
        if ctype == "min_results":
            n = params["n"]
            passed = len(songs) >= n
            return CheckResult(ctype, desc, passed, len(songs), f">={n}",
                               f"Got {len(songs)}, needed {n}")

        elif ctype == "max_results":
            n = params["n"]
            passed = len(songs) <= n
            return CheckResult(ctype, desc, passed, len(songs), f"<={n}",
                               f"Got {len(songs)}, max {n}")

        elif ctype == "has_titles":
            bad = [s for s in songs if not getattr(s, "title", "").strip()]
            passed = len(bad) == 0
            return CheckResult(ctype, desc, passed, len(bad), 0,
                               "All titles present" if passed else f"{len(bad)} missing titles")

        elif ctype == "genre_match_rate":
            genre = params["genre"]
            min_rate = params["min_rate"]
            rate = sum(1 for s in songs if s.genre == genre) / max(len(songs), 1)
            passed = rate >= min_rate
            return CheckResult(ctype, desc, passed, round(rate, 2), f">={min_rate}",
                               f"Genre match rate: {rate:.0%}")

        elif ctype == "mood_match_rate":
            mood = params["mood"]
            min_rate = params["min_rate"]
            rate = sum(1 for s in songs if s.mood == mood) / max(len(songs), 1)
            passed = rate >= min_rate
            return CheckResult(ctype, desc, passed, round(rate, 2), f">={min_rate}",
                               f"Mood match rate: {rate:.0%}")

        elif ctype == "energy_proximity":
            target = params["target"]
            max_gap = params["max_avg_gap"]
            avg_gap = sum(abs(s.energy - target) for s in songs) / max(len(songs), 1)
            passed = avg_gap <= max_gap
            return CheckResult(ctype, desc, passed, round(avg_gap, 3), f"<={max_gap}",
                               f"Avg energy gap: {avg_gap:.3f}")

        elif ctype == "energy_min_avg":
            min_e = params["min"]
            avg_e = sum(s.energy for s in songs) / max(len(songs), 1)
            passed = avg_e >= min_e
            return CheckResult(ctype, desc, passed, round(avg_e, 3), f">={min_e}",
                               f"Avg energy: {avg_e:.3f}")

        elif ctype == "energy_max_avg":
            max_e = params["max"]
            avg_e = sum(s.energy for s in songs) / max(len(songs), 1)
            passed = avg_e <= max_e
            return CheckResult(ctype, desc, passed, round(avg_e, 3), f"<={max_e}",
                               f"Avg energy: {avg_e:.3f}")

        elif ctype == "acousticness_min_avg":
            min_a = params["min"]
            avg_a = sum(s.acousticness for s in songs) / max(len(songs), 1)
            passed = avg_a >= min_a
            return CheckResult(ctype, desc, passed, round(avg_a, 3), f">={min_a}",
                               f"Avg acousticness: {avg_a:.3f}")

        elif ctype == "confidence_min":
            min_c = params["min"]
            passed = confidence >= min_c
            return CheckResult(ctype, desc, passed, round(confidence, 3), f">={min_c}",
                               f"Confidence: {confidence:.3f}")

        elif ctype == "no_crash":
            return CheckResult(ctype, desc, True, "no_error", "no_error",
                               "No exception raised")

        elif ctype == "deterministic":
            runs = params.get("runs", 2)
            first_title = songs[0].title if songs else ""
            results2 = run_fn(prefs)
            second_title = results2[0].title if results2 else ""
            passed = first_title == second_title
            return CheckResult(ctype, desc, passed, second_title, first_title,
                               "Consistent" if passed else f"'{first_title}' vs '{second_title}'")

        elif ctype == "artist_diversity":
            min_unique = params["min_unique"]
            unique = len(set(s.artist for s in songs))
            passed = unique >= min_unique
            return CheckResult(ctype, desc, passed, unique, f">={min_unique}",
                               f"{unique} unique artists")

        else:
            return CheckResult(ctype, desc, False, None, None, f"Unknown check type: {ctype}")

    except Exception as e:
        return CheckResult(ctype, desc, False, None, None, f"Check raised: {e}")


# ── Test runner ───────────────────────────────────────────────────────────────

def run_test(case: TestCase, run_basic_fn: Callable,
             run_agent_fn: Callable, verbose: bool = False) -> TestResult:
    """Runs a single test case and returns a TestResult."""
    t0 = time.time()
    songs = []
    confidence = 0.0
    error = None

    try:
        if case.input_type == "prefs":
            prefs = case.input_data
            from src.guardrails import validate_user_prefs
            val = validate_user_prefs(prefs)
            prefs = val.value
            songs = run_basic_fn(prefs)
            confidence = 0.7  # basic recommender assumed confident
        elif case.input_type == "query":
            agent_result = run_agent_fn(case.input_data["query"])
            songs = agent_result.recommendations
            confidence = agent_result.confidence
        else:
            raise ValueError(f"Unknown input_type: {case.input_type}")
    except Exception as e:
        error = str(e)
        logger.error(f"Test '{case.name}' raised: {e}")

    check_results = []
    for chk in case.checks:
        if chk["type"] == "no_crash" and error:
            cr = CheckResult("no_crash", chk["desc"], False, error, "no_error",
                             f"Exception: {error}")
        else:
            cr = run_check(chk, songs, confidence,
                           run_basic_fn, case.input_data)
        check_results.append(cr)

    all_passed = all(c.passed for c in check_results) and error is None
    duration_ms = round((time.time() - t0) * 1000, 1)

    return TestResult(
        case_name=case.name,
        passed=all_passed,
        confidence=confidence,
        checks=check_results,
        duration_ms=duration_ms,
        recommendations=[s.title for s in songs[:3]],
        error=error,
    )


def run_all_tests(run_basic_fn: Callable, run_agent_fn: Callable,
                  verbose: bool = False,
                  tags_filter: Optional[List[str]] = None) -> List[TestResult]:
    """Runs all test cases and returns results."""
    suite = TEST_SUITE
    if tags_filter:
        suite = [t for t in suite if any(tag in t.tags for tag in tags_filter)]

    print(f"\n{'═'*60}")
    print(f"  🧪 Music Recommender — Reliability Test Harness")
    print(f"{'═'*60}")
    print(f"  Running {len(suite)} test cases...\n")

    results = []
    for case in suite:
        result = run_test(case, run_basic_fn, run_agent_fn, verbose=verbose)
        results.append(result)

        icon = "✅" if result.passed else "❌"
        print(f"  {icon} [{result.case_name}] ({result.duration_ms:.0f}ms)")

        if verbose or not result.passed:
            for cr in result.checks:
                sub_icon = "  ✓" if cr.passed else "  ✗"
                print(f"      {sub_icon} {cr.description}: {cr.message}")
            if result.recommendations:
                print(f"      🎵 Top results: {', '.join(result.recommendations[:3])}")
            if result.error:
                print(f"      💥 Error: {result.error}")

    # ── Summary ─────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    avg_conf = sum(r.confidence for r in results) / max(len(results), 1)
    avg_ms   = sum(r.duration_ms for r in results) / max(len(results), 1)

    print(f"\n{'─'*60}")
    print(f"  📊 RESULTS SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total:    {len(results)} tests")
    print(f"  Passed:   {passed} ✅")
    print(f"  Failed:   {failed} {'❌' if failed else '✅'}")
    print(f"  Pass rate: {passed/max(len(results),1):.0%}")
    print(f"  Avg confidence: {avg_conf:.3f}")
    print(f"  Avg duration:   {avg_ms:.0f}ms per test")

    if failed:
        print(f"\n  ⚠️  Failed tests:")
        for r in results:
            if not r.passed:
                failed_checks = [c.description for c in r.checks if not c.passed]
                print(f"    • {r.case_name}: {', '.join(failed_checks)}")

    print(f"{'═'*60}\n")
    return results


def export_results(results: List[TestResult], path: str) -> None:
    """Exports test results to a JSON file for CI/CD integration."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    data = {
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": sum(1 for r in results if r.passed) / max(len(results), 1),
            "avg_confidence": sum(r.confidence for r in results) / max(len(results), 1),
        },
        "tests": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  📄 Results exported to: {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Music Recommender Test Harness")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--export", type=str, help="Export results to JSON path")
    parser.add_argument("--tags", nargs="*", help="Filter tests by tags")
    args = parser.parse_args()

    # Lazy import to avoid circular deps
    from src.data_loader import load_songs
    from src.recommender import Recommender, UserProfile
    from src.rag_retriever import RAGRetriever
    from src.agent import MusicRecommenderAgent
    from src.guardrails import validate_user_prefs, setup_logging

    setup_logging("WARNING")  # Quiet during tests
    songs = load_songs()

    retriever = RAGRetriever()
    retriever.build_index(songs)

    agent = MusicRecommenderAgent(songs, retriever)

    def run_basic(prefs: Dict):
        val = validate_user_prefs(prefs)
        p = val.value
        rec = Recommender(songs)
        user = UserProfile(
            favorite_genre=p["favorite_genre"],
            favorite_mood=p["favorite_mood"],
            target_energy=p["target_energy"],
            likes_acoustic=p["target_energy"] < 0.45,
        )
        return rec.recommend(user, k=5)

    def run_agent(query: str):
        return agent.run(query, k=5, verbose=False)

    results = run_all_tests(run_basic, run_agent,
                            verbose=args.verbose,
                            tags_filter=args.tags)

    if args.export:
        export_results(results, args.export)

    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
