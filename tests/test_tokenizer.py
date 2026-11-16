"""Tokenizer: round-trip, vocab, unknown chars."""

from app.services.tokenizer_service import CharTokenizer


def test_round_trip():
    """Encode then decode returns the original text."""
    tok = CharTokenizer.from_texts(["hello world", "why did the"])
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text


def test_vocab_is_sorted_unique():
    """The vocab is the sorted set of characters in the texts."""
    tok = CharTokenizer.from_texts(["bbaa", "cc"])
    assert tok.vocab == ["a", "b", "c"]
    assert tok.vocab_size == 3


def test_unknown_chars_skipped():
    """Characters outside the vocab are skipped, not crashed on."""
    tok = CharTokenizer(["a", "b"])
    assert tok.encode("axb") == [0, 1]
