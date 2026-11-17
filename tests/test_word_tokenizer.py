"""Word tokenizer: vocab, round-trip, eos, detokenizing."""

from app.services.tokenizer_service import (
    CharTokenizer,
    WordTokenizer,
    word_tokenize,
)


def test_word_tokenize_splits_words_and_punctuation():
    """Words, numbers, and punctuation are separate tokens."""
    assert word_tokenize("Why did the chicken cross the road?") == [
        "Why",
        "did",
        "the",
        "chicken",
        "cross",
        "the",
        "road",
        "?",
    ]
    assert word_tokenize("I have 2 cats!") == ["I", "have", "2", "cats", "!"]


def test_vocab_is_most_frequent_with_specials_first():
    """Vocab is [<pad>, <unk>, <eos>] then tokens by frequency."""
    tok = WordTokenizer.from_texts(
        ["the cat sat", "the dog sat", "the cat ran"], max_vocab=6
    )
    assert tok.vocab[:3] == ["<pad>", "<unk>", "<eos>"]
    assert tok.vocab[3] == "the"  # most frequent
    assert tok.vocab_size == 6
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.eos_id == 2


def test_unknown_words_become_unk():
    """A word outside the vocab maps to <unk>, not a crash."""
    tok = WordTokenizer.from_texts(["hello world"], max_vocab=10)
    ids = tok.encode("hello zzz")
    assert ids[0] == tok.stoi["hello"]
    assert ids[1] == tok.unk_id


def test_encode_jokes_appends_eos_per_joke():
    """Each joke ends with <eos> in the training stream."""
    tok = WordTokenizer.from_texts(["a b", "c d"], max_vocab=10)
    ids = tok.encode_jokes(["a b", "c d"])
    assert ids == [
        tok.stoi["a"],
        tok.stoi["b"],
        tok.eos_id,
        tok.stoi["c"],
        tok.stoi["d"],
        tok.eos_id,
    ]


def test_round_trip_reattaches_punctuation():
    """Encode then decode returns the original sentence."""
    tok = WordTokenizer.from_texts(
        ["Why did the chicken cross the road? To get to the other side!"],
        max_vocab=100,
    )
    text = "Why did the chicken cross the road? To get to the other side!"
    assert tok.decode(tok.encode(text)) == text


def test_decode_handles_quotes_and_brackets():
    """Quotes and brackets attach to the right side."""
    tok = WordTokenizer.from_texts(
        ['He said "hello" (loudly).'], max_vocab=100
    )
    assert tok.decode(tok.encode('He said "hello" (loudly).')) == (
        'He said "hello" (loudly).'
    )


def test_start_and_stop_are_eos():
    """Word generation starts from and stops at <eos>."""
    tok = WordTokenizer.from_texts(["a b"], max_vocab=10)
    assert tok.start_token_id == tok.eos_id
    assert tok.stop_token_id == tok.eos_id


def test_char_tokenizer_start_and_stop_are_newline():
    """The char tokenizer still starts/stops at a newline (v1)."""
    tok = CharTokenizer.from_texts(["a\nb"])
    assert tok.start_token_id == tok.stoi["\n"]
    assert tok.stop_token_id == tok.stoi["\n"]


def test_char_tokenizer_without_newline_never_stops():
    """If the vocab has no newline, stop_token_id is None — not id 0.

    Id 0 is a real token (a space in v1's vocab). Stopping on it
    would cut every sample at its first space; v1 samples always
    ran to max_tokens instead.
    """
    tok = CharTokenizer.from_texts(["ab cd"])
    assert "\n" not in tok.vocab
    assert tok.stop_token_id is None
    assert tok.start_token_id == 0  # v1's fallback: start from id 0
