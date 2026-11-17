"""Tokenizers: character-level (v1) and word-level (v2).

Both share the same interface so training and generation are
tokenizer-agnostic: ``from_texts`` / ``from_file`` / ``save``,
``vocab_size``, ``encode`` / ``decode``, ``encode_jokes`` (encode a
list of jokes into one training stream), and ``start_token_id`` /
``stop_token_id`` (the token generation starts from and stops at).

Character level: the vocab is the sorted corpus charset; jokes are
joined with newlines, so generation starts from and stops at "\\n".
Word level: the vocab is the most frequent words/punctuation plus
three specials; each joke ends with an <eos> token, so generation
starts from and stops at <eos>.
"""

import json
import re
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9']")


def word_tokenize(text: str) -> list[str]:
    """Split text into word/punctuation tokens, keeping case."""
    return WORD_RE.findall(text)


class CharTokenizer:
    """Maps characters to ids; vocab is the sorted corpus charset."""

    kind = "char"

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

    def encode_jokes(self, texts: list[str]) -> list[int]:
        """Join jokes with newlines into one training stream."""
        return self.encode("\n".join(texts) + "\n")

    @property
    def start_token_id(self) -> int:
        """Generation starts from a newline (the end of a joke).

        Falls back to id 0 when the vocab has no newline — that is
        what v1 did, so v1's sampling distribution is unchanged.
        """
        return self.stoi.get("\n", 0)

    @property
    def stop_token_id(self) -> int | None:
        """Generation stops at a newline — or never, if the vocab
        has no newline (v1's vocab drops it, so v1 samples always
        run to max_tokens). Returning None (not id 0) matters: id 0
        is a real token (a space), and stopping on it would cut
        every sample at its first space."""
        return self.stoi.get("\n")


class WordTokenizer:
    """Maps words/punctuation to ids; vocab is the most frequent tokens.

    The vocab is [<pad>, <unk>, <eos>] followed by the most frequent
    tokens in the corpus. Unknown words map to <unk>; every joke ends
    with <eos>, which is also the token generation starts from.
    """

    kind = "word"
    PAD = "<pad>"
    UNK = "<unk>"
    EOS = "<eos>"

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for i, tok in enumerate(self.vocab)}

    @classmethod
    def from_texts(
        cls, texts: list[str], max_vocab: int = 10000
    ) -> "WordTokenizer":
        """Build from the most frequent tokens (deterministic order)."""
        counts = Counter()
        for text in texts:
            counts.update(word_tokenize(text))
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        keep = [tok for tok, _ in top[: max(0, max_vocab - 3)]]
        return cls([cls.PAD, cls.UNK, cls.EOS] + keep)

    @classmethod
    def from_file(cls, path: Path) -> "WordTokenizer":
        """Load a saved vocab list."""
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        """Save the vocab list as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.vocab), encoding="utf-8")

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS]

    @property
    def start_token_id(self) -> int:
        """Generation starts from <eos> (the end of a joke)."""
        return self.eos_id

    @property
    def stop_token_id(self) -> int:
        """Generation stops at <eos> (the end of a joke)."""
        return self.eos_id

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Text to token ids (unknown words become <unk>)."""
        ids = [self.stoi.get(tok, self.unk_id) for tok in word_tokenize(text)]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_jokes(self, texts: list[str]) -> list[int]:
        """Encode each joke and append <eos> to it."""
        ids: list[int] = []
        for text in texts:
            ids.extend(self.encode(text, add_eos=True))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Token ids back to text, with punctuation reattached."""
        tokens = [self.itos[i] for i in ids if i in self.itos]
        text = ""
        in_quote = False
        prev_no_space = True  # no leading space at the start
        for tok in tokens:
            if tok in (self.EOS, self.PAD):
                continue
            if tok == self.UNK:
                chunk, space_before, no_space_after = "<unk>", True, False
            elif tok == '"':
                if in_quote:
                    chunk, space_before, no_space_after = '"', False, False
                    in_quote = False
                else:
                    chunk, space_before, no_space_after = '"', True, True
                    in_quote = True
            elif tok in {".", ",", "!", "?", ";", ":", "%", ")", "]", "}"}:
                chunk, space_before, no_space_after = tok, False, False
            elif tok in "([{":
                chunk, space_before, no_space_after = tok, True, True
            else:
                chunk, space_before, no_space_after = tok, True, False
            if text and space_before and not prev_no_space:
                text += " "
            text += chunk
            prev_no_space = no_space_after
        return text.strip()
