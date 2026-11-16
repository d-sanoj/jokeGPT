"""Automated quality signals for generated jokes.

These are cheap, honest signals — not a judge of funniness:
- complete: ends with terminal punctuation and is within length bounds
- duplicate: share of samples that repeat an earlier sample (normalised)
- mean length in characters and words
"""

import re

from app.models.generate import QualitySignals

MIN_CHARS = 20
MAX_CHARS = 300
TERMINAL = (".", "!", "?")


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace — for duplicate detection."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def quality_signals(samples: list[str]) -> QualitySignals:
    """Compute quality signals for a batch of generated jokes."""
    n = len(samples)
    if n == 0:
        return QualitySignals(
            n_samples=0,
            complete_pct=0.0,
            duplicate_pct=0.0,
            mean_chars=0.0,
            mean_words=0.0,
        )
    complete = sum(
        1
        for s in samples
        if MIN_CHARS <= len(s) <= MAX_CHARS and s.endswith(TERMINAL)
    )
    seen: set[str] = set()
    duplicates = 0
    for s in samples:
        norm = _normalise(s)
        if norm in seen:
            duplicates += 1
        seen.add(norm)
    lengths = [len(s) for s in samples]
    words = [len(s.split()) for s in samples]
    return QualitySignals(
        n_samples=n,
        complete_pct=round(100 * complete / n, 1),
        duplicate_pct=round(100 * duplicates / n, 1),
        mean_chars=round(sum(lengths) / n, 1),
        mean_words=round(sum(words) / n, 1),
    )
