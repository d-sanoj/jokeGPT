# jokeGPT — findings

The short version: **the model learned to spell, not to tell jokes.**
After 1,500 training steps on 15,000 jokes, the loss fell by more than
half (2.77 → 2.06) and the output looks like English at a glance —
real words, question marks, the odd "Why did the…" — but read any
sample to the end and it falls apart into misspelled fragments. Out of
50 samples from the finished model, only 2 (4%) ended with proper
punctuation and read as a finished sentence. None of the 50 is a joke
a person would recognise as one. Everything below is the evidence for
that, and what would have to change.

## What was built

- **Corpus:** 15,000 short jokes from the public Short Jokes dataset
  (Hugging Face `ysharma/short_jokes`, MIT licence — a mirror of the
  Kaggle Short Jokes set, 231,657 jokes). Cleaned: deduped, dropped
  empties and very short ones, filtered strong slurs and explicit
  terms (best effort — the source is edgy Reddit humour, so the filter
  is imperfect), 1,395,354 characters, 95-character vocabulary.
  See `data/README.md`.
- **Model:** character-level GPT (nanoGPT-style). Token + positional
  embeddings, 4 transformer blocks, final LayerNorm, output head tied
  to the token embedding. `d_model=128, n_layer=4, n_head=4`,
  context 128, **819,840 parameters**. AdamW, cosine LR, cross-entropy,
  95/5 train/val split.
- **Machine:** ~1.5 shared CPUs, ~6 GB RAM, no GPU. All wall-clock
  times below are on that box.

## The numbers

| run | steps | params | wall time | train loss | val loss | complete* | duplicates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main (full data) | 1500 | 819,840 | 33 min 38 s | 2.070 | **2.064** | 4% | 0% |
| main, early ckpt (step 500) | 500 | 819,840 | — | 2.462 | 2.445 | 8% | 0% |
| data-1k (1,000 jokes) | 500 | 819,840 | 8 min 34 s | 2.541 | 2.539 | 2% | 0% |
| small (d_model 64, 2 layers) | 500 | 113,856 | 1 min 47 s | 2.704 | 2.698 | 6% | 0% |

\* "Complete" = a sample that ends with terminal punctuation (`.`, `!`,
`?`) and is 20–300 characters long, out of 50 samples at temperature 0.8.
"Duplicates" = share of the 50 samples that repeat another sample.

Temperature sweep (same final checkpoint, 50 samples each):

| temperature | complete | duplicates |
| --- | --- | --- |
| 0.5 | 6% | 0% |
| 0.8 | 4% | 0% |
| 1.2 | 4% | 0% |

The loss curve is in `visuals/loss.png`. It falls fast for the first
200 steps (2.77 → 2.65), then grinds down slowly to 2.06 by step 1500
and is still falling, but flattening — about 0.02 per 100 steps at the
end.

## What helped

- **More data.** Same model, same 500 steps: 15,000 jokes gives val
  loss 2.445; 1,000 jokes gives 2.539. Fifteen times the data is worth
  about 0.1 of loss, and the 1k model finished only 2% of its jokes
  against the full-data model's 8% at the same step count.
- **More training.** The main model improved from val 2.445 at step
  500 to 2.064 at step 1500 — a real, steady gain, and the samples
  visibly improve too: step 500 is mostly letter soup ("tre aing whe
  ay turod she the no os owicner"), step 1500 has real words and
  sentence shapes, even if the sentences don't hold together.
- **A bigger model.** The small model (113,856 params) was the worst
  of the three 500-step runs at val 2.698. Capacity matters, even at
  this scale.

## What broke

- **The model cannot tell a joke.** This is the headline. Every
  ablation, every temperature, every prompt tried — the output is
  word-shaped noise. It learned the *look* of jokes (short lines,
  question marks, "What do you…", "Why did the…") without the
  *structure*. A loss of 2.06 sounds good next to the starting 4.55,
  but for a character-level model it still means the model is very
  unsure about each next letter, and that uncertainty is what you read
  as garbled words.
- **Loss is not readability.** The loss fell by more than half and the
  samples still aren't jokes. Anyone watching only the loss curve
  would think the model was working. It wasn't — not for this job.
- **Temperature doesn't rescue it.** Cold (0.5) gives 6% complete,
  hot (1.2) gives 4% — the knob barely matters, because the problem
  is in the model, not the sampling. Duplicates are 0% everywhere:
  the model never repeats itself, it just wanders.
- **Greedy decoding collapses.** Sampling the single most likely next
  character every time produces "the say the so the so the the some
  the the some…" — the model falls into a loop on its favourite word.
  Prompting helps the opening only: "Why did the chicken be is a can
  a bun their way so like that I can and a hapher say in the of
  herist…" — it starts like a joke and immediately loses the thread.

## Best and worst jokes (verbatim model output)

These are the least-bad and most-bad of the 50 samples from the
finished model, temperature 0.8, unedited. "Best" here means the most
recognisable English — none of them is a working joke.

**Best five:**

1. "the sblacte. Well Me Fight? OSt and with is with no that Gades, toy that fingit? Phoweest? A slean spine to becwion? Why"
2. "the got shirt goet the ex in that mith Neop you man her wom the was thim word 26! Down wher lught is me on atoze? Reenis"
3. "is a restie.I saying shid!! A clame If muth eall and livets.Wo. \"I'm for one with funt the tire I is youw narmarin was o"
4. "weencall and on a repeoned was the worked.What a cammele, I'm do It betled to clase nout a was titen a my reedy? The wor"
5. "I Rechimpene?\"What do you for the preated diferekentited healve maly one fish catourite? Me saide of a can tecentingy ch"

**Worst five:**

1. "ather sbarted his nivens you shrom I idd a band to mee some and the pust her an for hout abor* she so, Docking Illbay th"
2. "upione \"I the beill groodng reetts make have cifur this canther evind thinks ofter a gire: I Thane have to morth goold l"
3. "gistars, sare at and oweki baut 15 sonly and the odno have to cable was as the the doing oup arescecause tum Fabiting of"
4. "merdars main a be it upe resteaple to you wealt in the the bechiry heald go the the got, you hereny dood ek.I of corstri"
5. "the you have my recade like and blade the a workeang thing i\" clitted.\" \"GodayinHoure bampind of hesiding of les dogs.."

## Surprises

- **The early checkpoint "beat" the final one on the complete-joke
  metric (8% vs 4%).** That's noise, not a real effect: with only 50
  samples, the difference is 2 samples. The final model is better on
  loss by a wide margin and its words are visibly more real. A small
  sample makes a weak metric jump around — worth remembering before
  trusting any single percentage in this report.
- **The model learned joke openings.** "What do you…", "Why did
  the…", "How do you…" appear again and again, and often in the right
  place. It learned *where* a joke starts. It just can't get to the
  punchline.
- **No duplicates at all.** A model this weak could easily have
  memorised one or two jokes and repeated them. It didn't — 0%
  duplicates in every run. The outputs are all different, and all
  broken in different ways.

## What would fix it

Three things, in order of how much they would help:

1. **Word-level tokens instead of characters.** Predicting whole words
   instead of single letters is a much easier job, and the same size
   of model would produce readable sentences far sooner. This is the
   biggest lever and the obvious next experiment.
2. **Much more training.** The loss was still falling at step 1500.
   Ten times the steps would help, but on this box that is hours, and
   the gains per step are shrinking — it would improve the spelling
   more than the joke structure.
3. **A bigger model.** The small model was clearly worse, so capacity
   helps — but a bigger model on this CPU is slower per step, and
   capacity alone doesn't teach joke structure.

The honest summary: a 0.8M-parameter character-level GPT, trained for
~34 minutes on 15,000 jokes, learns the shape of joke text and not
the jokes. The experiment worked as an experiment — the comparisons
are clean and the numbers are real — but the model itself is a
starting point, not a joke teller.
