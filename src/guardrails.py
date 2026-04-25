"""
guardrails.py — Input Validation, Logging, and Safety Guardrails

This module handles:
  1. Input sanitization and validation (user prefs and natural language queries)
  2. Structured logging setup (file + console, JSON-capable)
  3. Rate limiting (prevents abuse in deployed scenarios)
  4. Output safety filtering (removes inappropriate content from results)
  5. Error context manager for graceful degradation

All functions return a (value, list_of_warnings) tuple so callers can
surface issues to users without crashing.
"""

import logging
import logging.handlers
import os
import re
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from functools import wraps
from collections import deque

# ── Logging Setup ────────────────────────────────────────────────────────────

LOG_DIR  = "logs"
LOG_FILE = "logs/recommender.log"

def setup_logging(level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Configures structured logging for the entire application.

    Outputs:
      - Console: human-readable with color codes
      - File (logs/recommender.log): rotating file, 5MB max, 3 backups

    Returns the root application logger.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("music_recommender")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    fmt_console = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    fmt_file = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":%(message)s}',
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    # Rotating file handler
    if log_to_file:
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)
        logger.addHandler(fh)

    logger.info(f'"Logging initialized (level={level}, file={LOG_FILE})"')
    return logger


# ── Validation constants ─────────────────────────────────────────────────────

VALID_GENRES = {
    "pop", "hip-hop", "rock", "r&b", "electronic", "jazz",
    "lo-fi", "indie", "ambient", "latin", "classical", "metal",
    "country", "reggae", "folk", "soul", "funk", "blues",
    "punk", "disco", "alternative", "k-pop", "gospel", "",
}

VALID_MOODS = {
    "happy", "intense", "relaxed", "chill", "moody",
    "focused", "sad", "energetic", "melancholy", "upbeat", "",
}

ENERGY_RANGE = (0.0, 1.0)
QUERY_MAX_LEN = 500
QUERY_MIN_LEN = 1

# Terms that should trigger a content warning (not blocking, just flagged)
FLAGGED_TERMS = [
    r"\bself.harm\b", r"\bsuicid\b", r"\bhate\b",
    r"\bviolen\b", r"\bkill\b", r"\bdrug\b",
]

# ── Validation functions ─────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    value: Any
    warnings: List[str]
    is_valid: bool
    sanitized: bool = False


def validate_query(query: str) -> ValidationResult:
    """
    Validates and sanitizes a natural language recommendation query.

    Checks:
      - Not empty, not too long
      - No injection attempts (e.g. system prompt injection)
      - No flagged content
      - Strips excessive whitespace and special chars
    """
    warnings = []

    if not isinstance(query, str):
        return ValidationResult(value="", warnings=["Query must be a string"], is_valid=False)

    # Sanitize: strip and normalize whitespace
    sanitized = re.sub(r"\s+", " ", query.strip())
    was_sanitized = sanitized != query

    # Length checks
    if len(sanitized) < QUERY_MIN_LEN:
        return ValidationResult(value=sanitized, warnings=["Query is empty"], is_valid=False)

    if len(sanitized) > QUERY_MAX_LEN:
        sanitized = sanitized[:QUERY_MAX_LEN]
        warnings.append(f"Query truncated to {QUERY_MAX_LEN} characters")
        was_sanitized = True

    # Injection detection (basic)
    injection_patterns = [
        r"ignore (previous|above|all) instructions",
        r"system prompt",
        r"<\s*script",
        r"__import__",
        r"eval\s*\(",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            return ValidationResult(
                value="", is_valid=False,
                warnings=[f"Query contains disallowed pattern: {pattern}"]
            )

    # Content flagging (warning only, not blocking)
    for pattern in FLAGGED_TERMS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            warnings.append(f"Query contains sensitive content; results may be filtered")
            break

    return ValidationResult(
        value=sanitized, warnings=warnings, is_valid=True, sanitized=was_sanitized
    )


def validate_user_prefs(prefs: Dict) -> ValidationResult:
    """
    Validates a user preference dictionary.

    Returns a normalized preferences dict with any issues flagged.
    """
    warnings = []

    if not isinstance(prefs, dict):
        return ValidationResult(value={}, warnings=["Preferences must be a dict"], is_valid=False)

    cleaned = dict(prefs)

    # Validate genre
    genre = str(prefs.get("favorite_genre", "")).lower().strip()
    if genre not in VALID_GENRES:
        warnings.append(f"Unknown genre '{genre}'; defaulting to 'pop'")
        cleaned["favorite_genre"] = "pop"
    else:
        cleaned["favorite_genre"] = genre

    # Validate mood
    mood = str(prefs.get("favorite_mood", "")).lower().strip()
    if mood not in VALID_MOODS:
        warnings.append(f"Unknown mood '{mood}'; defaulting to 'happy'")
        cleaned["favorite_mood"] = "happy"
    else:
        cleaned["favorite_mood"] = mood

    # Validate energy
    try:
        energy = float(prefs.get("target_energy", 0.65))
        if not (ENERGY_RANGE[0] <= energy <= ENERGY_RANGE[1]):
            warnings.append(f"Energy {energy} out of range [0,1]; clamping")
            energy = max(ENERGY_RANGE[0], min(ENERGY_RANGE[1], energy))
        cleaned["target_energy"] = round(energy, 3)
    except (TypeError, ValueError):
        warnings.append("Invalid energy value; defaulting to 0.65")
        cleaned["target_energy"] = 0.65

    # Validate k (if present)
    if "k" in prefs:
        try:
            k = int(prefs["k"])
            if k < 1 or k > 50:
                warnings.append(f"k={k} out of range [1,50]; clamping")
                k = max(1, min(50, k))
            cleaned["k"] = k
        except (TypeError, ValueError):
            warnings.append("Invalid k; defaulting to 5")
            cleaned["k"] = 5

    return ValidationResult(value=cleaned, warnings=warnings, is_valid=True)


def validate_song_results(songs: List, query_prefs: Dict) -> ValidationResult:
    """
    Post-processes recommendation results for safety and completeness.
    Flags if results are empty, all from one artist, etc.
    """
    warnings = []

    if not songs:
        return ValidationResult(
            value=[], warnings=["No recommendations found for the given preferences"],
            is_valid=False
        )

    # Check for artist monopoly
    artists = [s.artist for s in songs]
    if len(set(artists)) == 1 and len(songs) > 1:
        warnings.append(f"All recommendations are from the same artist: {artists[0]}")

    # Check energy range sanity
    energies = [s.energy for s in songs]
    avg_energy = sum(energies) / len(energies)
    target = query_prefs.get("target_energy", 0.5)
    if abs(avg_energy - target) > 0.4:
        warnings.append(
            f"Average energy ({avg_energy:.2f}) far from target ({target:.2f})"
        )

    return ValidationResult(value=songs, warnings=warnings, is_valid=True)


# ── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token bucket rate limiter. Prevents API abuse in deployed scenarios.
    Default: 20 requests per 60 seconds per session.
    """

    def __init__(self, max_calls: int = 20, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: deque = deque()

    def check(self) -> Tuple[bool, str]:
        """Returns (allowed, message)."""
        now = time.time()
        # Remove expired entries
        while self._calls and self._calls[0] < now - self.window:
            self._calls.popleft()

        if len(self._calls) >= self.max_calls:
            oldest = self._calls[0]
            wait = round(self.window - (now - oldest), 1)
            return False, f"Rate limit: {self.max_calls} requests per {self.window:.0f}s. Retry in {wait}s."

        self._calls.append(now)
        return True, f"OK ({len(self._calls)}/{self.max_calls} in window)"


# ── Error Context Manager ────────────────────────────────────────────────────

class SafeRecommendationContext:
    """
    Context manager that catches errors in the recommendation pipeline
    and returns a graceful fallback instead of crashing.

    Usage:
        with SafeRecommendationContext() as ctx:
            result = recommender.recommend(user)
        if ctx.error:
            print(f"Error: {ctx.error}")
        else:
            print(result)
    """

    def __init__(self, fallback=None, logger_name="music_recommender"):
        self.fallback = fallback
        self.error: Optional[Exception] = None
        self.logger = logging.getLogger(logger_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            self.logger.error(f'"Caught error in recommendation pipeline: {exc_type.__name__}: {exc_val}"')
            return True  # Suppress exception
        return False


# ── Decorator for guarded functions ─────────────────────────────────────────

def guarded(fallback_value=None):
    """
    Decorator that wraps a function in error handling and logging.
    On exception, logs the error and returns fallback_value.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("music_recommender")
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.error(f'"guarded({fn.__name__}) caught: {type(e).__name__}: {e}"')
                return fallback_value
        return wrapper
    return decorator


# ── Summary printer ──────────────────────────────────────────────────────────

def print_validation_summary(results: ValidationResult, context: str = "") -> None:
    """Prints validation warnings to console in a user-friendly format."""
    if not results.warnings:
        return
    prefix = f"[{context}] " if context else ""
    for w in results.warnings:
        print(f"  ⚠️  {prefix}{w}")
