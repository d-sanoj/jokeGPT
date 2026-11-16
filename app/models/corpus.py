"""Corpus schemas."""

from pydantic import BaseModel


class Joke(BaseModel):
    """One joke from the cleaned corpus."""

    text: str
    source: str


class CorpusStats(BaseModel):
    """Corpus summary: size, length, vocabulary."""

    jokes: int
    chars: int
    mean_joke_chars: float
    vocab_size: int
    vocab: str
    source: str
