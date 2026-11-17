#!/usr/bin/env python3
"""Run the jokeGPT v2 upgrade sequence and record the results.

One lever at a time, measured after each step:
  step1   word-level tokens (same 15k jokes, same-ish model)
  decode  decoding sweep on the step-1 checkpoint (temp / top-k / top-p)
  step3   more data: 50k jokes from the same cleaned pool
  step4   bigger model: d_model 256, 6 layers (old plan)
  vocab-data  step 2 (user's call): 40k vocab + 50k jokes
  big-model   step 3 (user's call): bigger model, same 40k vocab + 50k
  step5   longer training on the best combination
  final   combine the best of each (only if 80% not yet reached)

Good% is a human judgment (complete + coherent + joke-shaped, 50
samples per step) — this script saves the samples verbatim and leaves
good_pct empty; judge them afterwards with:
  PYTHONPATH=. python scripts/run_v2.py judge <run_id> <verdicts.json>
where verdicts.json is a list of 50 objects with keys
text / complete / coherent / joke_shaped / good / note.

Training runs are skipped when their final checkpoint already exists,
so a re-run after a failure does not redo finished work.
Results are written to experiments/runs-v2.json; samples live in
experiments/v2-samples/<run_id>.json (and decode settings in
experiments/v2-samples/decode-<label>.json).
Usage: PYTHONPATH=. python scripts/run_v2.py step1
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from app.core.logging import configure_logging
from app.models.train import ModelConfig
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.evaluation_service import quality_signals
from app.services.generation_service import GenerationService
from app.services.training_service import TrainingService

HERE = Path(__file__).resolve().parent.parent
RUNS_V2 = HERE / "experiments" / "runs-v2.json"
SAMPLES_DIR = HERE / "experiments" / "v2-samples"

N_SAMPLES = 50
GEN_SEED = 123
GEN_TEMP = 0.8
GEN_TOP_K = 40


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _params_from_config(config: dict) -> int:
    """Parameter count from a config dict (accounts for weight tying)."""
    v, d = config["vocab_size"], config["d_model"]
    n, ctx = config["n_layer"], config["context_len"]
    per_block = 3 * d * d + d * d + 8 * d * d + 4 * d
    return v * d + ctx * d + n * per_block + 2 * d


def _load_records() -> list[dict]:
    if not RUNS_V2.exists():
        return []
    return json.loads(RUNS_V2.read_text(encoding="utf-8")).get("runs", [])


def _save_records(records: list[dict]) -> None:
    RUNS_V2.parent.mkdir(parents=True, exist_ok=True)
    RUNS_V2.write_text(
        json.dumps({"runs": records}, indent=2), encoding="utf-8"
    )


def _save_samples(run_id: str, samples: list[str], meta: dict) -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / f"{run_id}.json"
    path.write_text(
        json.dumps({"run_id": run_id, **meta, "samples": samples}, indent=2),
        encoding="utf-8",
    )
    return path


def _finish_run(training, generation, checkpoints, config, record):
    """Generate 50 samples + quality signals for a finished run."""
    samples = generation.generate_batch(
        N_SAMPLES,
        temperature=GEN_TEMP,
        top_k=GEN_TOP_K,
        checkpoint=config.run_id,
        seed=GEN_SEED,
    )
    record["samples"] = samples
    record["quality"] = quality_signals(samples).model_dump()
    record["good_pct"] = None  # human judgment, filled in by `judge`
    record["final_checkpoint"] = str(
        checkpoints.checkpoint_path(config.run_id, config.max_steps)
    )
    samples_path = _save_samples(
        config.run_id,
        samples,
        {
            "temperature": GEN_TEMP,
            "top_k": GEN_TOP_K,
            "top_p": 1.0,
            "seed": GEN_SEED,
        },
    )
    record["samples_file"] = str(samples_path)
    log(
        f"{config.run_id}: quality complete="
        f"{record['quality']['complete_pct']}% duplicate="
        f"{record['quality']['duplicate_pct']}% "
        f"mean_chars={record['quality']['mean_chars']} "
        f"samples -> {samples_path}"
    )
    return record


def _train_step(
    training,
    generation,
    checkpoints,
    config,
    step_no: int,
    what_changed: str,
):
    """Train one step (resumable) and record it in runs-v2.json."""
    records = _load_records()
    final = checkpoints.checkpoint_path(config.run_id, config.max_steps)
    if final.exists():
        log(f"{config.run_id}: final checkpoint exists, skipping training")
        payload = checkpoints.load_checkpoint(final)
        history = payload.get("loss_history", [])
        record = {
            "run_id": config.run_id,
            "step": step_no,
            "what_changed": what_changed,
            "label": config.run_id,
            "config": config.model_dump(),
            "train_loss": history[-1]["train_loss"] if history else None,
            "val_loss": history[-1]["val_loss"] if history else None,
            "wall_seconds": 0.0,
            "steps": config.max_steps,
            "params": _params_from_config(payload["config"]),
            "samples": [],
            "loss_history": history,
            "quality": None,
            "good_pct": None,
            "checkpoints": [str(final)],
        }
    else:
        latest = checkpoints.latest_checkpoint(config.run_id)
        if latest is not None:
            log(f"{config.run_id}: resuming from {latest}")
            record = training.train(config, resume_from=latest)
        else:
            log(f"{config.run_id}: training {config.max_steps} steps")
            record = training.train(config)
        record["step"] = step_no
        record["what_changed"] = what_changed
    record = _finish_run(training, generation, checkpoints, config, record)
    records = [r for r in records if r["run_id"] != config.run_id]
    records.append(record)
    _save_records(records)
    log(f"{config.run_id}: recorded (val {record['val_loss']:.3f})")
    return record


def cmd_step1():
    """Step 1: word-level tokens, same 15k jokes, same-ish model."""
    configure_logging()
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    generation = GenerationService(checkpoints)
    config = ModelConfig(
        run_id="word-level",
        tokenizer="word",
        max_vocab=10000,
        context_len=64,
        d_model=128,
        n_layer=4,
        n_head=4,
        max_steps=1500,
        eval_every=100,
        checkpoint_every=500,
    )
    _train_step(
        training,
        generation,
        checkpoints,
        config,
        step_no=1,
        what_changed=(
            "word-level tokens (10k vocab) instead of characters; "
            "context 64 words; same 15k jokes, same model size"
        ),
    )
    log("step 1 done")


def cmd_decode():
    """Step 2: decoding sweep on the word-level checkpoint."""
    configure_logging()
    checkpoints = CheckpointRepository()
    generation = GenerationService(checkpoints)
    if checkpoints.latest_checkpoint("word-level") is None:
        print("no word-level checkpoint yet — run step1 first")
        sys.exit(1)
    settings = [
        ("temp-0.5", {"temperature": 0.5, "top_k": 40, "top_p": 1.0}),
        ("temp-1.0", {"temperature": 1.0, "top_k": 40, "top_p": 1.0}),
        ("temp-1.2", {"temperature": 1.2, "top_k": 40, "top_p": 1.0}),
        ("topk-10", {"temperature": 0.8, "top_k": 10, "top_p": 1.0}),
        ("topp-0.9", {"temperature": 0.8, "top_k": 40, "top_p": 0.9}),
    ]
    rows = []
    for label, kwargs in settings:
        samples = generation.generate_batch(
            N_SAMPLES, checkpoint="word-level", seed=GEN_SEED, **kwargs
        )
        quality = quality_signals(samples).model_dump()
        path = _save_samples(
            f"decode-{label}",
            samples,
            {"checkpoint": "word-level", "seed": GEN_SEED, **kwargs},
        )
        rows.append(
            {"label": label, **kwargs, "quality": quality, "good_pct": None}
        )
        log(
            f"decode {label}: complete={quality['complete_pct']}% "
            f"duplicate={quality['duplicate_pct']}% samples -> {path}"
        )
    out = HERE / "experiments" / "v2-decoding.json"
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    log(f"wrote {out} (baseline temp-0.8/top-k-40 is the step-1 record)")
    log("step 2 done — judge each decode-<label>.json, then pick the best")


def _ensure_50k_corpus() -> Path:
    """Build data/jokes-50k.jsonl if it does not exist yet."""
    path = HERE / "data" / "jokes-50k.jsonl"
    if path.exists():
        return path
    log("building 50k-joke corpus (same shuffle, so it contains v1's 15k)")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "clean_corpus.py"),
            "--sample-n",
            "50000",
            "--out",
            str(path),
        ],
        check=True,
    )
    return path


def cmd_vocab_data():
    """Step 2 (user's call, 2026-11-16): bigger vocab + more data.

    The decoding sweep is deferred, not cancelled. Step 1 showed the
    10k vocab cap is the binding constraint (44/50 samples contain
    <unk>), so this step attacks it directly: 40k vocab (99.59% of
    corpus tokens, 0.41% <unk>) and 50k jokes from the same cleaned
    pool. 40k was picked over 30k (98.65% / 1.35% <unk>) because
    step 1 shows the model amplifies the training <unk> rate ~3x in
    generation (4.6% of training tokens -> <unk> in 88% of samples):
    at 30k that still leaves roughly a third of samples with <unk>,
    capping good% well below the 80% target, while 40k should cut
    it to ~15%. Smoke-tested first: 40k runs at 4.67 s/step, peak
    RSS 1.76 GB (vs 3.52 s/step, 1.48 GB at 30k) — slower and
    heavier, but it ran clean, so 40k is the largest tractable size.
    2500 steps = ~5 epochs over the 50k corpus.
    """
    configure_logging()
    corpus_path = _ensure_50k_corpus()
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    generation = GenerationService(checkpoints)
    config = ModelConfig(
        run_id="word-level-50k",
        tokenizer="word",
        max_vocab=40000,
        context_len=64,
        d_model=128,
        n_layer=4,
        n_head=4,
        max_steps=2500,
        eval_every=250,
        checkpoint_every=1000,
        jokes_file=str(corpus_path),
    )
    _train_step(
        training,
        generation,
        checkpoints,
        config,
        step_no=2,
        what_changed=(
            "bigger vocab (40k words, 99.59% corpus coverage) + 50k "
            "jokes (same cleaned pool, contains v1's 15k); decoding "
            "sweep deferred by user choice; word-level, d_model 128"
        ),
    )
    log("step 2 (vocab+data) done")


def cmd_step3():
    """Step 3: more data — 50k jokes, word-level, same model."""
    configure_logging()
    corpus_path = _ensure_50k_corpus()
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    generation = GenerationService(checkpoints)
    config = ModelConfig(
        run_id="word-level-50k",
        tokenizer="word",
        max_vocab=10000,
        context_len=64,
        d_model=128,
        n_layer=4,
        n_head=4,
        max_steps=2500,
        eval_every=250,
        checkpoint_every=1000,
        jokes_file=str(corpus_path),
    )
    _train_step(
        training,
        generation,
        checkpoints,
        config,
        step_no=3,
        what_changed=(
            "50k jokes (same cleaned pool, contains v1's 15k), "
            "2500 steps, word-level"
        ),
    )
    log("step 3 done")


def cmd_big_model():
    """Step 3 (user's call, 2026-11-16): bigger model.

    The user picked the bigger model over the deferred decoding sweep.
    Step 2's loss flattened late (5.29 -> 5.26 over the last 750
    steps), so capacity, not steps, is the bottleneck at d_model 128.
    This step changes ONLY capacity: d_model 256, 6 layers, 8 heads,
    context 96 words — same 40k vocab (99.59% coverage) and the same
    50k jokes, so the comparison with step 2 is clean. Smoke-tested
    first (scripts/smoke_big.py): 12.33 s/step, peak RSS 2.84 GB —
    1500 steps ≈ 5.1h, the same ~4.4-epoch exposure as step 2, chosen
    to finish ~00:30 CST with a wide buffer before the 06:49 blog run.
    """
    configure_logging()
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    generation = GenerationService(checkpoints)
    config = ModelConfig(
        run_id="word-level-50k-big",
        tokenizer="word",
        max_vocab=40000,
        context_len=96,
        d_model=256,
        n_layer=6,
        n_head=8,
        max_steps=1500,
        eval_every=250,
        checkpoint_every=500,
        jokes_file=str(HERE / "data" / "jokes-50k.jsonl"),
    )
    _train_step(
        training,
        generation,
        checkpoints,
        config,
        step_no=3,
        what_changed=(
            "bigger model: d_model 256, 6 layers, 8 heads, context 96 "
            "words; same 40k vocab and 50k jokes as step 2 — only "
            "capacity changes; decoding sweep still deferred"
        ),
    )
    log("step 3 (bigger model) done")


def cmd_step4():
    """Step 4: bigger model — d_model 256, 6 layers, word-level."""
    configure_logging()
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    generation = GenerationService(checkpoints)
    config = ModelConfig(
        run_id="word-level-big",
        tokenizer="word",
        max_vocab=10000,
        context_len=64,
        d_model=256,
        n_layer=6,
        n_head=8,
        max_steps=1500,
        eval_every=100,
        checkpoint_every=500,
    )
    _train_step(
        training,
        generation,
        checkpoints,
        config,
        step_no=4,
        what_changed=(
            "bigger model: d_model 256, 6 layers, 8 heads; "
            "word-level, same 15k jokes"
        ),
    )
    log("step 4 done")


def cmd_step5(from_run: str, steps: int):
    """Step 5: longer training on the best combination so far."""
    configure_logging()
    records = _load_records()
    base = next((r for r in records if r["run_id"] == from_run), None)
    if base is None:
        print(f"no v2 record for {from_run!r} — run that step first")
        sys.exit(1)
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    generation = GenerationService(checkpoints)
    config = ModelConfig(**base["config"])
    config.run_id = f"{from_run}-long"
    config.max_steps = steps
    _train_step(
        training,
        generation,
        checkpoints,
        config,
        step_no=5,
        what_changed=(
            f"longer training: {from_run} config extended to {steps} steps"
        ),
    )
    log("step 5 done")


def cmd_final(from_run: str, steps: int):
    """Final: combine the best of each lever, if 80% not yet reached."""
    configure_logging()
    records = _load_records()
    base = next((r for r in records if r["run_id"] == from_run), None)
    if base is None:
        print(f"no v2 record for {from_run!r}")
        sys.exit(1)
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    generation = GenerationService(checkpoints)
    config = ModelConfig(**base["config"])
    config.run_id = "word-level-best"
    config.max_steps = steps
    _train_step(
        training,
        generation,
        checkpoints,
        config,
        step_no=6,
        what_changed=(
            f"final combination: best config ({from_run}) with {steps} steps"
        ),
    )
    log("final run done")


def cmd_judge(run_id: str, verdicts_path: Path):
    """Merge human verdicts into a run record.

    verdicts.json: a list of 50 objects with keys text / complete /
    coherent / joke_shaped / good / note. The texts must match the
    saved samples for the run.
    """
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    if len(verdicts) != N_SAMPLES:
        print(f"expected {N_SAMPLES} verdicts, got {len(verdicts)}")
        sys.exit(1)
    records = _load_records()
    record = next((r for r in records if r["run_id"] == run_id), None)
    if record is None:
        print(f"no v2 record for {run_id!r}")
        sys.exit(1)
    saved = record.get("samples", [])
    for verdict, sample in zip(verdicts, saved):
        if verdict["text"] != sample:
            print("verdict texts do not match the saved samples")
            sys.exit(1)
    good = sum(1 for v in verdicts if v["good"])
    record["good_pct"] = round(100 * good / len(verdicts), 1)
    record["judgement"] = {
        "good": good,
        "n": len(verdicts),
        "complete": sum(1 for v in verdicts if v["complete"]),
        "coherent": sum(1 for v in verdicts if v["coherent"]),
        "joke_shaped": sum(1 for v in verdicts if v["joke_shaped"]),
    }
    out = SAMPLES_DIR / f"{run_id}-verdicts.json"
    out.write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    _save_records(records)
    log(
        f"{run_id}: good {good}/{len(verdicts)} "
        f"({record['good_pct']}%) — verdicts -> {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("step1", help="step 1: word-level tokens")
    sub.add_parser("decode", help="step 2: decoding sweep")
    sub.add_parser(
        "vocab-data",
        help="step 2 (user's call): bigger vocab + more data",
    )
    sub.add_parser("step3", help="step 3: more data (50k jokes)")
    sub.add_parser("step4", help="step 4: bigger model (old plan)")
    sub.add_parser(
        "big-model",
        help="step 3 (user's call): bigger model, same 40k vocab + 50k jokes",
    )
    p5 = sub.add_parser("step5", help="step 5: longer training")
    p5.add_argument("--from", dest="from_run", required=True)
    p5.add_argument("--steps", type=int, required=True)
    pf = sub.add_parser("final", help="final: combine the best")
    pf.add_argument("--from", dest="from_run", required=True)
    pf.add_argument("--steps", type=int, required=True)
    pj = sub.add_parser("judge", help="merge human verdicts into a record")
    pj.add_argument("run_id")
    pj.add_argument("verdicts", type=Path)
    args = parser.parse_args()
    if args.command == "step1":
        cmd_step1()
    elif args.command == "decode":
        cmd_decode()
    elif args.command == "vocab-data":
        cmd_vocab_data()
    elif args.command == "step3":
        cmd_step3()
    elif args.command == "step4":
        cmd_step4()
    elif args.command == "big-model":
        cmd_big_model()
    elif args.command == "step5":
        cmd_step5(args.from_run, args.steps)
    elif args.command == "final":
        cmd_final(args.from_run, args.steps)
    elif args.command == "judge":
        cmd_judge(args.run_id, args.verdicts)


if __name__ == "__main__":
    main()
