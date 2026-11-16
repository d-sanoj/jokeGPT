"""Load the cleaned jokes corpus and describe it."""

import json
import random

from app.core.config import settings
from app.core.logging import get_logger
from app.models.corpus import CorpusStats, Joke

logger = get_logger(__name__)


class CorpusRepository:
    """Reads data/jokes.jsonl; the corpus is bundled with the repo."""

    def __init__(self, jokes_file=None):
        self.jokes_file = jokes_file or settings.jokes_file
        self._jokes: list[Joke] | None = None

    def load_jokes(self) -> list[Joke]:
        """Load all jokes (cached after the first read)."""
        if self._jokes is None:
            jokes = []
            with self.jokes_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        jokes.append(Joke(**json.loads(line)))
            self._jokes = jokes
            logger.info("loaded %d jokes from %s", len(jokes), self.jokes_file)
        return self._jokes

    def stats(self) -> CorpusStats:
        """Corpus summary: size, length, vocabulary, source."""
        jokes = self.load_jokes()
        texts = [j.text for j in jokes]
        chars = sum(len(t) for t in texts)
        vocab = "".join(sorted(set("".join(texts))))
        source = jokes[0].source if jokes else ""
        return CorpusStats(
            jokes=len(jokes),
            chars=chars,
            mean_joke_chars=chars / max(1, len(jokes)),
            vocab_size=len(vocab),
            vocab=vocab,
            source=source,
        )

    def train_val_split(self, seed: int = 42, train_split: float = 0.95):
        """Split jokes into train/val lists (shuffled, seeded)."""
        jokes = list(self.load_jokes())
        random.Random(seed).shuffle(jokes)
        cut = int(len(jokes) * train_split)
        return jokes[:cut], jokes[cut:]

    def sample(self, n: int, seed: int = 42) -> list[Joke]:
        """A seeded random subset of the corpus (for ablations)."""
        jokes = list(self.load_jokes())
        random.Random(seed).shuffle(jokes)
        return jokes[:n]
