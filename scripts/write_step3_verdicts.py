#!/usr/bin/env python3
"""Write the step-3 verdicts file from the judged samples.

Verdicts are (complete, coherent, joke_shaped, note) in sample order.
complete follows the automated rule (20-300 chars, ends . ! ?);
coherent/joke_shaped are the human read against experiments/judging-rubric.md.
good = complete AND coherent AND joke_shaped.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SAMPLES = HERE / "experiments" / "v2-samples" / "word-level-50k-big.json"
OUT = HERE / "experiments" / "v2-samples" / "word-level-50k-big-verdicts.json"

VERDICTS = [
    (False, False, False, "Contains <unk>; no terminal punctuation."),
    (True, True, False, "Grammatical; a meta-statement, not a joke."),
    (
        False,
        False,
        False,
        "Ends with a quote; 'I asked me if I would get the time "
        "the street' is garbled.",
    ),
    (
        True,
        False,
        False,
        "The 'When' clause never resolves into a sensible sentence.",
    ),
    (
        True,
        False,
        False,
        "Classic setup, but the answer is word salad; no punchline.",
    ),
    (
        True,
        False,
        False,
        "'who have a small way to the door' is garbled; no answer given.",
    ),
    (False, False, False, "Ends with a quote; garbled."),
    (
        True,
        False,
        False,
        "Setup and answer both garbled ('had a joke for a lot of dead').",
    ),
    (
        False,
        False,
        False,
        "No terminal punctuation; 'Because it was the last night' "
        "is not a punchline.",
    ),
    (False, False, False, "Ends with a quote; garbled."),
    (True, False, False, "'for the way of them a man' is garbled."),
    (
        False,
        False,
        False,
        "No terminal punctuation; 'A man' is not an answer.",
    ),
    (False, False, False, "Ends with a quote; garbled."),
    (
        True,
        False,
        False,
        "Nested question never answered; 'A man is a tree and a "
        "new tree and a woman' is word salad.",
    ),
    (True, False, False, "Contains <unk>."),
    (True, False, False, "Contains <unk>."),
    (True, False, False, "'If you have a lot like to make a cat' is garbled."),
    (True, False, False, "Contains <unk>; fragment."),
    (
        False,
        False,
        False,
        "No terminal punctuation; 'He has a black guy' is not a punchline.",
    ),
    (True, False, False, "'watch a new girl in my friend' is garbled."),
    (
        True,
        False,
        False,
        "'The old date is all the people in the end' is word salad.",
    ),
    (True, False, False, "Word salad."),
    (
        True,
        False,
        False,
        "'I was a bad' is a fragment; the 'When' clause doesn't compose.",
    ),
    (
        True,
        False,
        False,
        "'the opposite of an hour and an American' is garbled; "
        "'It can't even' is a fragment.",
    ),
    (False, False, False, "Ends with a quote; trails off mid-sentence."),
    (True, False, False, "Garbled throughout."),
    (False, False, False, "Ends with a quote; number soup."),
    (True, False, False, "Contains <unk>."),
    (False, False, False, "No terminal punctuation; garbled answer."),
    (False, False, False, "Contains <unk>; ends with '*'; pure noise."),
    (
        True,
        False,
        False,
        "Echoes step 2's best sample but worse: "
        "'A cat, a man' is not a punchline.",
    ),
    (
        True,
        False,
        False,
        "Closest to a real joke: the classic light-bulb format, "
        "but the punchline is mangled ('the other day' for 'the "
        "light bulb') and 'How many dog' is ungrammatical.",
    ),
    (True, False, False, "Garbled."),
    (False, False, False, "Contains <unk>; no terminal punctuation."),
    (
        True,
        False,
        False,
        "'he made at the doctor' and 'I was a terrible hand' are garbled.",
    ),
    (
        True,
        False,
        False,
        "'that don't have with an octopus' is ungrammatical; "
        "'we have a good at their body' is garbled.",
    ),
    (True, False, False, "Word salad."),
    (
        True,
        False,
        False,
        "'What does the difference between a Mexican guy with 2?' "
        "is garbled; 'A woman had a lightbulb' is not a punchline.",
    ),
    (False, False, False, "No terminal punctuation; garbled."),
    (False, False, False, "Ends with a quote; garbled."),
    (True, False, False, "'I was a man in my friend' is garbled; fragment."),
    (
        True,
        False,
        False,
        "'I had a little restaurant from my wife "
        "told me to have a guy' is garbled.",
    ),
    (
        True,
        False,
        False,
        "'who is to say to the time' and 'he's a lot on' are garbled.",
    ),
    (True, True, False, "Grammatical; 'In the wrong' is not a punchline."),
    (True, False, False, "Contains <unk> twice."),
    (
        True,
        False,
        False,
        "Contains <unk>; 'before he cross' is ungrammatical.",
    ),
    (True, False, False, "'that one of a great person' is garbled."),
    (
        True,
        False,
        False,
        "'the first girl on the guy that his car "
        "is being on the wrong day' is garbled.",
    ),
    (
        True,
        True,
        False,
        "Grammatical; 'The first guy who got in her house' "
        "is not a punchline.",
    ),
    (True, True, False, "Grammatical; 'You do' is not a punchline."),
]


def main() -> None:
    data = json.loads(SAMPLES.read_text())
    samples = data["samples"]
    assert len(samples) == 50, f"expected 50 samples, got {len(samples)}"
    assert len(VERDICTS) == 50, f"expected 50 verdicts, got {len(VERDICTS)}"
    out = []
    for text, (complete, coherent, joke_shaped, note) in zip(
        samples, VERDICTS
    ):
        # Cross-check complete against the automated rule.
        auto = 20 <= len(text) <= 300 and text.endswith((".", "!", "?"))
        assert complete == auto, (
            f"complete mismatch on {text!r}: mine {complete}, auto {auto}"
        )
        out.append(
            {
                "text": text,
                "complete": complete,
                "coherent": coherent,
                "joke_shaped": joke_shaped,
                "good": complete and coherent and joke_shaped,
                "note": note,
            }
        )
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    n_complete = sum(v["complete"] for v in out)
    n_coherent = sum(v["coherent"] for v in out)
    n_shaped = sum(v["joke_shaped"] for v in out)
    n_good = sum(v["good"] for v in out)
    n_unk = sum(1 for v in out if "<unk>" in v["text"])
    print(
        f"wrote {OUT.name}: complete {n_complete}/50, "
        f"coherent {n_coherent}/50, joke-shaped {n_shaped}/50, "
        f"good {n_good}/50, <unk> in {n_unk}/50 samples"
    )


if __name__ == "__main__":
    main()
