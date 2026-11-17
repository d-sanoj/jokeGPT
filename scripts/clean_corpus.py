#!/usr/bin/env python3
"""Clean the short-jokes corpus into data/jokes.jsonl.

Source: Hugging Face ysharma/short_jokes (MIT), a mirror of the Kaggle
Short Jokes dataset (231,657 user-submitted jokes, mostly from Reddit).

Steps: parse CSV -> strip/collapse whitespace -> length filter (20-500
chars) -> drop mostly-non-ASCII -> profanity/slur filter (wordlist, best
effort — the source is user-submitted and edgy; filtering is imperfect)
-> dedupe on normalized text -> seeded sample of 15,000.

Writes data/jokes.jsonl: one joke per line {"text": ..., "source": ...}.
Prints a cleaning report to stdout.
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "shortjokes.csv"
OUT = HERE / "data" / "jokes.jsonl"
SOURCE = "ysharma/short_jokes (Kaggle Short Jokes, MIT)"
SAMPLE_N = 15000
SEED = 42
MIN_LEN = 20
MAX_LEN = 500

# Best-effort blocklist: strong slurs and explicit sexual terms. The source
# corpus is user-submitted jokes and contains edgy material; this list
# removes the worst of it but is not a guarantee of clean content.
BLOCKLIST = {
    "nigger",
    "nigga",
    "faggot",
    "dyke",
    "kike",
    "spic",
    "chink",
    "retard",
    "retarded",
    "cunt",
    "pussy",
    "dick",
    "cock",
    "blowjob",
    "handjob",
    "porn",
    "masturbat",
    "orgasm",
    "vagina",
    "penis",
    "raped",
    "rape",
    "molest",
    "pedophile",
    "paedophile",
    "shit",
    "fuck",
    "bitch",
    "whore",
    "slut",
}
WORD_RE = re.compile(r"[a-z]+")


def is_blocked(text: str) -> bool:
    words = set(WORD_RE.findall(text.lower()))
    return bool(words & BLOCKLIST)


def main() -> None:
    """Clean the raw jokes and write a seeded sample as JSONL.

    ``--sample-n`` / ``--out`` let later steps build a bigger sample
    from the same shuffle (same seed, so a 50k sample contains the
    same 15k jokes v1 used, plus 35k more).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-n", type=int, default=SAMPLE_N)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    sample_n, out, seed = args.sample_n, args.out, args.seed
    raw_rows = 0
    jokes: list[str] = []
    seen: set[str] = set()
    dropped = {
        "empty_or_short": 0,
        "too_long": 0,
        "non_ascii": 0,
        "blocked": 0,
        "duplicate": 0,
    }
    with RAW.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows += 1
            text = (row.get("Joke") or "").strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) < MIN_LEN:
                dropped["empty_or_short"] += 1
                continue
            if len(text) > MAX_LEN:
                dropped["too_long"] += 1
                continue
            ascii_chars = sum(1 for ch in text if ord(ch) < 128)
            if ascii_chars / max(1, len(text)) < 0.9:
                dropped["non_ascii"] += 1
                continue
            if is_blocked(text):
                dropped["blocked"] += 1
                continue
            norm = text.lower()
            if norm in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(norm)
            jokes.append(text)
    random.Random(seed).shuffle(jokes)
    sampled = jokes[:sample_n]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for text in sampled:
            f.write(json.dumps({"text": text, "source": SOURCE}) + "\n")
    chars = sum(len(t) for t in sampled)
    vocab = sorted(set("".join(sampled)))
    print(f"raw rows:            {raw_rows}")
    print(f"dropped:             {dropped}")
    print(f"cleaned (pre-sample): {len(jokes)}")
    print(f"sampled:             {len(sampled)}")
    print(f"total chars:         {chars:,}")
    print(f"mean joke length:    {chars / max(1, len(sampled)):.1f} chars")
    print(f"vocab size:          {len(vocab)} chars")
    print(f"vocab: {''.join(vocab)!r}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
