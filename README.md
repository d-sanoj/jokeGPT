# jokeGPT

A tiny GPT that tells jokes — trained from scratch, on CPU, as a
FastAPI service.

The model is a character-level transformer with about 0.8 million
parameters (not billions). It was trained on 15,000 short jokes and
learned the *shape* of joke text — real words, question marks, the
odd "Why did the…" — but it cannot yet tell a coherent joke. That is
the honest result, and it is the point of this repo: the code, the
trained weights, the measurements, and an account of what worked and
what did not. The full story is in `FINDINGS.md`.

## Quickstart

```bash
git clone https://github.com/d-sanoj/jokeGPT.git
cd jokeGPT
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app            # http://127.0.0.1:8000, docs at /docs
```

## The API

| Endpoint | What it does |
| --- | --- |
| `GET /health` | Liveness probe |
| `GET /api/v1/corpus` | Corpus stats (jokes, characters, vocabulary) |
| `POST /api/v1/train` | Start a training job (background), returns a job id |
| `GET /api/v1/train/{job_id}` | Job status: step, train/val loss, elapsed time |
| `POST /api/v1/generate` | Sample a joke from a checkpoint |
| `GET /api/v1/experiments` | All recorded experiment runs |
| `GET /api/v1/experiments/{run_id}` | One run: config, metrics, samples |

```bash
# What is in the corpus?
curl http://127.0.0.1:8000/api/v1/corpus

# Tell me a joke (from the trained main model)
curl -X POST http://127.0.0.1:8000/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"temperature": 0.8, "max_tokens": 120, "checkpoint": "main"}'

# Colder (safer, more repetitive) or hotter (wilder, more gibberish)
curl -X POST http://127.0.0.1:8000/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"temperature": 0.5, "max_tokens": 120, "checkpoint": "main"}'

# Start a training job (d_model 128, 4 layers, 1500 steps by default)
curl -X POST http://127.0.0.1:8000/api/v1/train \
  -H 'Content-Type: application/json' \
  -d '{"config": {"run_id": "mine", "max_steps": 500}}'

# Check on it
curl http://127.0.0.1:8000/api/v1/train/<job_id>

# The experiment results
curl http://127.0.0.1:8000/api/v1/experiments
```

## Replicating the experiment

```bash
# 1. Rebuild the corpus (downloads the public dataset, cleans it)
python scripts/clean_corpus.py

# 2. Run the full experiment suite (main model + ablations)
PYTHONPATH=. python scripts/run_experiments.py

# 3. Make the charts
PYTHONPATH=. python scripts/make_visuals.py

# 4. Run the tests
PYTHONPATH=. python -m pytest
```

The suite trains the main model (1500 steps) plus three ablations
and a temperature sweep, then writes the numbers to
`experiments/runs.json` and `experiments/temperature.json`. On a
small CPU box the whole suite takes about 50 minutes.
Expected headline numbers are in `FINDINGS.md`. The test suite
(`tests/`) covers the tokenizer, the model, the API, and a generation
smoke test that loads the trained main model and generates text.

## Project structure

```
app/                  the service (MVC)
  core/               settings (pydantic-settings), logging
  models/             Pydantic schemas: corpus, train, generate, experiment
  repositories/       corpus loading; checkpoint + run-record persistence
  services/           tokenizer, GPT model, training, generation, evaluation
  api/v1/             controllers: health, corpus, train, generate, experiments
  main.py             FastAPI app factory (python -m app to run)
data/                 jokes.jsonl (15,000 cleaned jokes) + README.md
checkpoints/          trained weights (main + ablations)
visuals/              charts for the blog post + the numbers behind them
experiments/          runs.json, temperature.json (recorded results)
scripts/              clean_corpus.py, run_experiments.py, make_visuals.py
tests/                unit + API tests + generation smoke test
```

## Architecture

MVC, the same layering as
[retriever-doc-answer](https://github.com/d-sanoj/retriever-doc-answer):
controllers in `app/api/v1/` are thin — they validate the request and
hand off to a service. Services hold the logic (the GPT, the training
loop, sampling, quality signals). Repositories own persistence (the
corpus file, checkpoint files, run records). Models are Pydantic
schemas for every request and response. Nothing in the service reads
outside the repo: the corpus and the weights are bundled.

## The model

Character-level GPT (nanoGPT-style): token embedding + learned
positional embedding, 4 transformer blocks (causal self-attention +
GELU MLP), final LayerNorm, output head tied to the token embedding.
Vocabulary is the 95 printable ASCII characters in the corpus.
Main config: `d_model=128`, `n_layer=4`, `n_head=4`, context 128,
819,840 parameters. Trained with AdamW, cosine learning-rate schedule,
cross-entropy on next-character prediction, 95/5 train/val split.

## Results

The headline: **the model learned to spell, not to tell jokes.**
Main run: val loss 2.064 (from 2.77 at step 100), but only 4% of 50
samples finished as a complete sentence, and none is a recognisable
joke. More data helped (15k jokes: val 2.445 vs 1k jokes: 2.539 at
500 steps), more training helped (val 2.445 → 2.064 from step 500 to
1500), and the small model was worst (val 2.698) — but none of it
produced a joke. Temperature barely mattered (complete 6% at 0.5,
4% at 0.8 and 1.2). See `FINDINGS.md` for the numbers, the ablations,
and the best and worst jokes the model actually produced.
