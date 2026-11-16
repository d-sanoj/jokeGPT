# Corpus

## Source

**Hugging Face `ysharma/short_jokes` (MIT license)**, a mirror of the
Kaggle *Short Jokes* dataset: 231,657 user-submitted short jokes,
mostly from Reddit. Download:

```
https://huggingface.co/datasets/ysharma/short_jokes
```

The raw CSV is not committed to this repo (24 MB); download it to
`data/raw/shortjokes.csv` and run `scripts/clean_corpus.py` to
reproduce `jokes.jsonl`.

## Cleaning (`scripts/clean_corpus.py`)

1. Parse the CSV (`ID`, `Joke`).
2. Collapse whitespace (including newlines) to single spaces.
3. Drop jokes shorter than 20 characters.
4. Drop jokes with more than 10% non-ASCII characters.
5. Drop jokes matching a blocklist of strong slurs and explicit
   sexual terms (best effort — the source is user-submitted and edgy;
   filtering is imperfect and the corpus may still contain jokes
   some readers find offensive).
6. Deduplicate on lowercased text.
7. Seeded shuffle (seed 42) and sample 15,000 jokes.

Result: 219,175 jokes survived cleaning; 15,000 sampled.
`jokes.jsonl` is one joke per line: `{"text": ..., "source": ...}`.

## Numbers

- Jokes: 15,000
- Characters: 1,395,354 (mean 93 per joke)
- Vocabulary: 95 distinct characters (printable ASCII)
