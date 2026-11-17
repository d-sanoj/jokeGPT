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

---

# v2 — the upgrade sequence

Target: **80% good samples** (40 of 50). "Good" is fixed for every
step and judged against the same rubric (`experiments/judging-rubric.md`):
a sample is good only if it is (1) complete — ends with `.` `!` or `?`
and is 20–300 characters, (2) coherent — grammatical English throughout,
no garbled fragments, no `<unk>` in the text, and (3) joke-shaped — has
a setup and a punchline, reads as a joke even if weak. All 50 samples
per step are saved verbatim in `experiments/v2-samples/` with their
verdicts, so the judgment is auditable. One lever at a time, one
commit per step. Stop early at 80%; stop if two consecutive levers
each gain under 5 points of good%.

Note on comparing losses across steps: v1's loss (2.06) and v2's loss
(5.03) are **not** on the same scale. Loss is measured against the
vocabulary, and v1's character vocab has 95 symbols (random guessing
= ln 95 ≈ 4.55) while v2's word vocab has 10,000 (random = ln 10000
≈ 9.21). The honest comparison between steps is good%, complete%,
and the samples themselves — not the raw loss number.

## Step 1 — word-level tokens

**What changed:** the tokenizer. Same 15,000 jokes, same-ish model
(d_model 128, 4 layers), but the model now predicts whole words
instead of single letters. Vocabulary: the 10,000 most frequent words
in the corpus plus `<pad>`, `<unk>`, `<eos>` — top-10k covers 95.4%
of corpus tokens, so 4.6% of training tokens are `<unk>`. Context 64
words (corpus mean 21.3 words/joke, p90 33, max 84). 2,079,488
parameters. Trained 1,500 steps, temp 0.8 / top-k 40, 50 samples,
fixed seeds 123+i.

**The numbers:**

| | v1 (char) | step 1 (word) |
| --- | --- | --- |
| val loss | 2.064 | 5.026 (train 4.738) |
| complete | 4% | **62%** (31/50) |
| good | 0% | **0%** (0/50) |
| coherent | — | 6% (3/50) |
| joke-shaped | — | 0% (0/50) |
| duplicates | 0% | 0% |
| mean sample | — | 59.6 chars, 12.8 words |

Wall time ~40 minutes total: the first attempt reached step 900
(val 5.076) before a service restart killed the process; training
resumed from the step-500 checkpoint and finished steps 501–1500 in
1,183 s. Resume support (`resume_from` in the training service, plus
a test) was added so a killed run no longer restarts from scratch.

**What it did:** word-level tokens fixed the spelling problem
completely — every sample is made of real words now, no letter soup.
Complete% jumped from 4% to 62%. The model also learned joke
*openings* even better than v1: "What do you call a…", "Why did
the…", "My wife…", "Knock…" appear in the right places, and the
samples have the rhythm of jokes.

**What broke:** `<unk>`. The model leans on the unknown-word token
as a filler — it appears in **44 of 50 samples**, often several
times per sample. The rubric fails any sample containing `<unk>` on
coherence, and rightly so: "<unk>" is not a word a person would read.
Even the 6 samples without `<unk>` are not jokes — they are plain,
slightly garbled statements with no setup and no punchline. So good%
stayed at 0% despite the big complete% jump. The model learned the
shape of a joke even more convincingly, and still cannot deliver one.

**Best samples (the 3 coherent ones — the least-bad of the 50):**

1. "When I guess I'll go over the head to be the other day."
2. "Did you mean you need? So you see a good joke!"
3. "I think I don't know this is not not good."

None is a joke. The first two are grammatical sentences with no
setup or punchline; the third is a garbled plain statement.

**Worst samples:**

1. "<unk>: \"<unk>\" I think he will have a few joke. \"to watch you\""
2. "What do you call a <unk> <unk>? <unk> <unk>"
3. "The <unk>, <unk> is the <unk>, I have a bar.................................. and the <unk>."
4. "I have to <unk>.............. <unk>......"
5. "What does the difference between a car and the difference between \"in a man who who were a <unk>?\" <unk>. \"It is there, it's a <unk>.\""

**What this means for the next steps:** the `<unk>` crutch is the
single biggest blocker, and it is a decoding problem as much as a
training problem — the model was trained with `<unk>` as a real
token, so at sampling time it reaches for it whenever it is unsure.
Step 2 (decoding sweep) will test whether colder sampling or a
different top-k/top-p keeps `<unk>` out, and whether forbidding
`<unk>` at generation time lets the model's real vocabulary do the
work instead. If decoding can't fix it, the fix is a bigger vocab or
no `<unk>` at all.

**Note (v1 quirk, found while building v2):** v1's character
vocabulary contains no newline, so v1's training stream was jokes
concatenated with no separator, and v1 generation never stopped
early — it always ran to 120 tokens. That is v1 behaviour and is
left as-is; v2's word tokenizer appends `<eos>` per joke and
generation stops at it.

## Step 2 — bigger vocab + more data (the user's call)

**What changed:** the decoding sweep was skipped on the user's
choice (2026-11-16) and this step went straight at the `<unk>`
problem instead. Two levers at once: the vocabulary grew from the
10,000 most frequent words to **40,000** (measured on the cleaned
50k corpus first — 40k covers 99.59% of corpus tokens, so only
0.41% of training tokens are `<unk>`; 30k would have covered
98.65% / 1.35%, and step 1 showed the model amplifies the training
`<unk>` rate ~3x in generation, so 30k would still have left
roughly a third of samples with `<unk>`), and the data grew from
15,000 to **50,000 jokes** from the same cleaned pool (the same
seeded shuffle, so the 50k contains v1's 15k). Same architecture
otherwise: d_model 128, 4 layers, context 64 words. 5,919,488
parameters (up from 2,079,488 — the embedding is most of it).
Trained 2,500 steps (~5 epochs over the 50k corpus), temp 0.8 /
top-k 40, 50 samples, same fixed seeds. Random-guessing loss for a
40k vocab is ln 40000 ≈ 10.6, so the loss numbers here are not on
step 1's scale (random ≈ 9.21) either.

**The numbers:**

| | v1 (char) | step 1 (word) | step 2 (word-50k) |
| --- | --- | --- | --- |
| val loss | 2.064 | 5.026 (train 4.738) | 5.262 (train 4.914) |
| complete | 4% | 62% (31/50) | **68%** (34/50) |
| good | 0% | 0% (0/50) | **0%** (0/50) |
| coherent | — | 6% (3/50) | **18%** (9/50) |
| joke-shaped | — | 0% (0/50) | **2%** (1/50) |
| samples with `<unk>` | — | 88% (44/50) | **12%** (6/50) |
| duplicates | 0% | 0% | 0% |
| mean sample | — | 59.6 chars, 12.8 words | 80.2 chars, 18.1 words |

Wall time ~2h20m of training: the first run reached step 2250
(val 5.250) before the process was killed with no error — a
runtime cleanup took out the training process and the monitoring
shells at the same time. It resumed from the step-2000 checkpoint
and finished steps 2001–2500 in 1,530 s. The resume restarts the
optimizer fresh, so the re-run step-2250 val (5.262) is a touch
above the pre-kill 5.250 — a negligible difference. Loss flattened
late: val 5.291 at step 1750, 5.267 at 2000, 5.262 at 2500 — more
steps at this size are not buying much.

**What it did:** the vocab lever worked exactly as designed.
`<unk>` collapsed from 44 of 50 samples to **6 of 50**, with just
6 `<unk>` tokens in total. Coherent samples tripled (3 → 9), the
samples got longer and more developed (18.1 words vs 12.8), and
for the first time a sample has a real punchline shape (see below).
Complete% crept up 62% → 68%.

**What broke:** good% is still **0%**. The blocker moved. The 9
coherent samples are all grammatical plain statements or
question-and-answer pairs with no twist — none reads as a joke.
And the one sample that does have a punchline shape is garbled in
the setup. So no sample is complete AND coherent AND joke-shaped
at the same time. The model can now spell, finish sentences, and
mostly avoid `<unk>` — but it still cannot put a setup and a
punchline together in one sample.

**Best samples (the 9 coherent ones — none is a joke):**

1. "How do you get a woman with me? I got a girl."
2. "How many fish are you with a Mexican? You know when I said I hate it."
3. "What is the difference between a man and a woman? A black man is a light bulb and two people."
4. "What did the new people say when he got the first time? A man has been a good person."
5. "Why are the black people for it? Because he is the first."
6. "My wife said he made the same house."
7. "What do you call a lot of a dog? A: The same night."
8. "Why was the first girl on the guy? Because she had to play the wrong."
9. "What do you call a long body? They got a group of her house."

**The one joke-shaped sample (garbled, so not good):**

1. "What do you cross a dog with sex with a light bulb? A baby."
   — the cross-breeding setup with "A baby" as the answer is a
   real punchline shape, the first in the whole sequence. But the
   setup mashes two "with" phrases together, so it fails
   coherence.

**Worst samples (the 6 with `<unk>` — the whole remaining `<unk>` problem):**

1. "What's the difference between a man and a man? To be a <unk>!"
2. "What does a new favorite man get that? A <unk>"
3. "[for <unk>] A: What does a lightbulb? Me: I make you? Me: It's pretty nice. Me: I'll come in my time. Me: "Oh I am.""
4. "I had a little than you have a long thing about in the time. What is the only one on the other? A man in the <unk>."
5. "My wife is the <unk> about his favorite thing they get back off from the other day."
6. "Who was a little woman to get a <unk> an entire time? The best thing about it is the last night in bed."

**What this means for the next steps:** vocabulary is no longer
the binding constraint — 12% of samples still hit `<unk>`, but
even a perfect vocab would only unlock the 9 coherent samples,
and none of those is a joke. The blocker is now composition: the
model needs to learn setup → punchline as a structure, not just
words. The most promising next levers are the bigger model (d_model
256, 6 layers — more capacity to hold the joke structure) and the
deferred decoding sweep (colder / top-p sampling might keep the
model on the joke-shaped track instead of drifting into plain
statements). Longer training on this size is the weakest lever —
val loss has flattened. If the bigger model still produces no
joke-shaped samples, the honest conclusion is that this
architecture needs either much more data or a different approach,
and the blog should say so.

## Step 3 — bigger model (the user's call)

**What changed:** one lever, capacity. d_model 128 → **256**,
n_layer 4 → **6**, n_head 4 → **8**, context 64 → **96 words**.
Everything else identical to step 2: the same 40k vocab (99.59%
coverage), the same 50,000 jokes, batch 32, lr 3e-4, temp 0.8 /
top-k 40, 50 samples, same fixed seeds. 14,997,504 parameters
(up from 5,919,488 — 2.5x). Trained 1,500 steps, which at context
96 is ~4.4 epochs over the 50k corpus — the same exposure as step
2's ~4.8 epochs, so the comparison is clean on data. Smoke-tested
first (`scripts/smoke_big.py`, 25 steps): 12.33 s/step, peak RSS
2.84 GB; the real run settled at ~8.7 s/step. Random-guessing loss
for a 40k vocab is ln 40000 ≈ 10.6, same scale as step 2.

**The numbers:**

| | v1 (char) | step 1 (word) | step 2 (word-50k) | step 3 (big) |
| --- | --- | --- | --- | --- |
| val loss | 2.064 | 5.026 (train 4.738) | 5.262 (train 4.914) | **5.225** (train 4.743) |
| complete | 4% | 62% (31/50) | 68% (34/50) | **70%** (35/50) |
| good | 0% | 0% (0/50) | 0% (0/50) | **0%** (0/50) |
| coherent | — | 6% (3/50) | 18% (9/50) | 8% (4/50) |
| joke-shaped | — | 0% (0/50) | 2% (1/50) | 0% (0/50) |
| samples with `<unk>` | — | 88% (44/50) | 12% (6/50) | 18% (9/50) |
| duplicates | 0% | 0% | 0% | 0% |
| mean sample | — | 59.6 chars, 12.8 words | 80.2 chars, 18.1 words | 79.4 chars, 18.0 words |

Wall time ~4.4h of training across three legs; 5h33m wall-clock
from launch (19:17 CST) to finish (00:50 CST). The runtime
restarted twice and killed the training process both times — once
at step 750 (val 5.255) and once at step 1250 (val 5.184). Each
time it resumed cleanly from the latest 500-step checkpoint (step
500, then step 1000) and finished, which is why checkpoints every
500 steps matter. The resume restarts the optimizer fresh, so the
re-run evals land a touch above the pre-kill ones (step 750: 5.329
vs 5.255; step 1250: 5.250 vs 5.184) — small, same direction both
times, so it is the resume, not noise. The recorded loss curve is
the redo curve: 5.755 (250) → 5.419 (500) → 5.329 (750) → 5.248
(1000) → 5.250 (1250) → 5.225 (1500). Loss flattened late again:
val 5.248 at step 1000, 5.250 at 1250, 5.225 at 1500.

**What it did:** val loss improved 5.262 → **5.225**, the best so
far — but only by 0.037, about the size of the resume effect, so
treat it as a small win, not a breakthrough. Complete% crept up
68% → 70%. The model reaches for the same joke templates as step
2 (light-bulb, cross-breeding, walks-into-a-bar) — the templates
are learned; the composition is not.

**What broke:** good% is still **0%**. Coherent samples fell 9 →
4 and joke-shaped 1 → 0. The bigger model is not composing better
jokes — it is composing slightly worse ones, or at best the same,
with 2.5x the parameters. The `<unk>` rate ticked up 6 → 9 samples
(noise-level, but it did not improve). The honest read: at this
scale, capacity was **not** the bottleneck. The loss says the
model predicts the next word a little better; the samples say it
still cannot put a setup and a punchline together in one sample.
More capacity bought a better loss number and nothing the rubric
can see.

**Best samples (the 4 coherent ones — none is a joke):**

1. "I'm going to make a joke about the day."
2. "What do you call a cow with a girl who gets married? In the wrong."
3. "What does a little Mexican say? The first guy who got in her house!"
4. "Who was a little woman in the end? You do."

**Closest to a joke (mangled punchline):**

1. "How many dog does it take to change a light paper? One to screw in the other day."
   — the classic light-bulb joke, with the punchline's last words
   swapped for "the other day". The model knows the shape; it
   cannot land it.

**Worst samples:**

1. "[for 2 year 3.. * takes a <unk> <unk> * * * * knock * * * * * * *"
   — pure noise.
2. "I got the best part of my ex - - - 1 / 11. 2.. 1: 1.. \"I didn't know what it is that one day!\""
   — number soup.
3. "What do you cross a dog with sex? A cat, a man........"
   — echoes step 2's best ("cross a dog with sex with a light
   bulb? A baby") but the answer is two nouns and trailing dots.

**What this means for the next steps:** capacity is ruled out as
the bottleneck at this scale — 2.5x the parameters bought 0.04 of
val loss and zero good jokes. The remaining levers in the
sequence: the deferred decoding sweep (colder / top-p sampling
might keep the model on the joke-shaped track — now the most
promising cheap lever, since `<unk>` is mostly gone and the
templates are there), more data (the full 219k cleaned jokes — the
one lever not yet pulled at full scale, and the model has now seen
the 50k set ~4.4 times), and longer training (weakest — val has
flattened twice). If the decoding sweep also lands at 0%, the
honest conclusion is that next-word prediction on this data, at
this scale, learns joke surface but not joke structure — and the
blog should say so.

The API now serves this checkpoint by default: `POST
/api/v1/generate` with no `checkpoint` field samples from
`word-level-50k-big` (step 1500, val 5.225) — the best checkpoint
so far. Any earlier run is still reachable by name
(`"main"`, `"word-level"`, `"word-level-50k"`).
