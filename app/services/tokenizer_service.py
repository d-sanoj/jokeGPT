"""Character-level tokenizer built from the corpus vocabulary."""

import json
from pathlib import Path


class CharTokenizer:
    """Maps characters to ids; vocab is the sorted corpus charset."""

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        """Build the tokenizer from a list of training texts."""
        vocab = sorted(set("".join(texts)))
        return cls(vocab)

    @classmethod
    def from_file(cls, path: Path) -> "CharTokenizer":
        """Load a saved vocab list."""
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        """Save the vocab list as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.vocab), encoding="utf-8")

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Text to token ids (unknown chars are skipped)."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        """Token ids back to text."""
        return "".join(self.itos[i] for i in ids if i in self.itos)
