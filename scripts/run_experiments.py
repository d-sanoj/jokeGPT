#!/usr/bin/env python3
"""Run the jokeGPT experiment suite and record the results.

Runs (wall-clock on this CPU box is reported for each):
  main      full corpus (15k jokes), d_model 128, 4 layers, 1500 steps
  data-1k   1k jokes, same model, 500 steps (vs main @ step 500)
  small     d_model 64, 2 layers, same corpus, 500 steps
  main-early  samples + loss from the main run's step-500 checkpoint
  temperature  generation at 0.5 / 0.8 / 1.2 from the main checkpoint

Training runs are skipped when their final checkpoint already exists,
so a re-run after a failure does not redo finished work.
Results are written to experiments/runs.json (training runs) and
experiments/temperature.json (the sampling sweep).
Usage: PYTHONPATH=. python scripts/run_experiments.py
"""

import json
import time

import torch

from app.core.logging import configure_logging
from app.models.train import ModelConfig
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.evaluation_service import quality_signals
from app.services.generation_service import GenerationService
from app.services.model_service import build_model
from app.services.tokenizer_service import CharTokenizer
from app.services.training_service import TrainingService

N_SAMPLES = 50
GEN_TEMP = 0.8


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def finish_run(service, generation, checkpoints, config, run):
    """Generate samples + quality signals for a finished run."""
    final_step = config.max_steps
    samples = generation.generate_batch(
        N_SAMPLES, temperature=GEN_TEMP, checkpoint=config.run_id, seed=123
    )
    run["samples"] = samples
    run["quality"] = quality_signals(samples).model_dump()
    run["final_checkpoint"] = str(
        checkpoints.checkpoint_path(config.run_id, final_step)
    )
    log(
        f"{config.run_id}: quality complete="
        f"{run['quality']['complete_pct']}% duplicate="
        f"{run['quality']['duplicate_pct']}% "
        f"mean_chars={run['quality']['mean_chars']}"
    )
    return run


def main():
    configure_logging()
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    generation = GenerationService(checkpoints)

    configs = [
        ModelConfig(
            run_id="main", max_steps=1500, eval_every=100, checkpoint_every=500
        ),
        ModelConfig(
            run_id="data-1k",
            max_steps=500,
            eval_every=100,
            checkpoint_every=500,
            data_subset=1000,
        ),
        ModelConfig(
            run_id="small",
            max_steps=500,
            eval_every=100,
            checkpoint_every=500,
            d_model=64,
            n_layer=2,
            n_head=2,
        ),
    ]
    records: list[dict] = []
    for config in configs:
        final = checkpoints.checkpoint_path(config.run_id, config.max_steps)
        if final.exists():
            log(f"{config.run_id}: final checkpoint exists, skipping training")
            # Rebuild the record from the checkpoint + loss history.
            payload = checkpoints.load_checkpoint(final)
            history = payload.get("loss_history", [])
            record = {
                "run_id": config.run_id,
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
                "checkpoints": [str(final)],
            }
        else:
            log(f"{config.run_id}: training {config.max_steps} steps")
            record = training.train(config)
        record = finish_run(training, generation, checkpoints, config, record)
        records.append(record)
        checkpoints.save_runs(records)
        log(f"{config.run_id}: recorded (val {record['val_loss']:.3f})")

    # (b) Training length: the main run's step-500 checkpoint.
    early_path = checkpoints.checkpoint_path("main", 500)
    if early_path.exists():
        payload = checkpoints.load_checkpoint(early_path)
        history = payload.get("loss_history", [])
        step500 = next((h for h in history if h["step"] == 500), None)
        samples = _generate_from(
            generation, early_path, N_SAMPLES, GEN_TEMP, 123
        )
        early = {
            "run_id": "main-early",
            "label": "main @ step 500",
            "config": payload["config"],
            "train_loss": step500["train_loss"] if step500 else None,
            "val_loss": step500["val_loss"] if step500 else None,
            "wall_seconds": 0.0,
            "steps": 500,
            "params": _params_from_config(payload["config"]),
            "samples": samples,
            "loss_history": [h for h in history if h["step"] <= 500],
            "quality": quality_signals(samples).model_dump(),
            "checkpoints": [str(early_path)],
        }
        records.append(early)
        checkpoints.save_runs(records)
        log(
            f"main-early: val {early['val_loss']:.3f}, "
            f"complete {early['quality']['complete_pct']}%"
        )

    # (d) Temperature sweep from the main final checkpoint.
    sweep = {"checkpoint": "main", "n_samples": N_SAMPLES, "rows": []}
    for temp in (0.5, 0.8, 1.2):
        samples = generation.generate_batch(
            N_SAMPLES, temperature=temp, checkpoint="main", seed=123
        )
        row = {
            "temperature": temp,
            "quality": quality_signals(samples).model_dump(),
            "samples": samples[:5],
        }
        sweep["rows"].append(row)
        log(
            f"temperature {temp}: complete="
            f"{row['quality']['complete_pct']}% duplicate="
            f"{row['quality']['duplicate_pct']}%"
        )
    out = checkpoints.experiments_file.parent / "temperature.json"
    out.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    log(f"wrote {out}")
    log("experiment suite done")


def _params_from_config(config: dict) -> int:
    """Parameter count from a config dict (accounts for weight tying)."""
    v, d = config["vocab_size"], config["d_model"]
    n, ctx = config["n_layer"], config["context_len"]
    # qkv (3d^2) + attn out (d^2) + mlp (8d^2) + norms (4d)
    per_block = 3 * d * d + d * d + 8 * d * d + 4 * d
    return v * d + ctx * d + n * per_block + 2 * d


def _generate_from(generation, path, n, temperature, seed):
    """Generate from one specific checkpoint file (not the latest)."""
    payload = generation.checkpoints.load_checkpoint(path)
    tokenizer = CharTokenizer(payload["vocab"])
    model = build_model(payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    out = []
    for i in range(n):
        gen = torch.Generator()
        gen.manual_seed(seed + i)
        ids = [tokenizer.stoi.get("\n", 0)]
        made = 0
        with torch.no_grad():
            while made < 120:
                ctx = torch.tensor([ids], dtype=torch.long)
                logits = model.generate_step(ctx)
                logits = logits[0] / temperature
                top_values, _ = torch.topk(logits, 40)
                logits[logits < top_values[-1]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=gen))
                ids.append(nxt)
                made += 1
                if tokenizer.itos[nxt] == "\n":
                    break
        out.append(tokenizer.decode(ids).strip())
    return out


if __name__ == "__main__":
    main()
