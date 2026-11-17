#!/usr/bin/env python3
"""Smoke-test the bigger-model config before the overnight run.

Trains 25 steps with the exact step-3 config (d_model 256, 6 layers,
context 96, 40k vocab, 50k jokes) under run_id "smoke-big" so its
checkpoint cannot be picked up by the real run. Reports s/step and
peak RSS, so max_steps can be sized to the overnight window.
Usage: TMPDIR=~/workspace/tmp PYTHONPATH=. python scripts/smoke_big.py
"""

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import configure_logging
from app.models.train import ModelConfig
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.training_service import TrainingService

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    configure_logging()
    corpus = CorpusRepository()
    checkpoints = CheckpointRepository()
    training = TrainingService(corpus, checkpoints)
    config = ModelConfig(
        run_id="smoke-big",
        tokenizer="word",
        max_vocab=40000,
        context_len=96,
        d_model=256,
        n_layer=6,
        n_head=8,
        max_steps=25,
        eval_every=25,
        checkpoint_every=1000,
        jokes_file=str(HERE / "data" / "jokes-50k.jsonl"),
    )
    t0 = time.time()
    record = training.train(config)
    wall = time.time() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(
        f"SMOKE big-model: {record['params']} params, "
        f"val {record['val_loss']:.3f}, {wall:.1f}s for 25 steps "
        f"({wall / 25:.2f} s/step), peak RSS {peak_gb:.2f} GB",
        flush=True,
    )
    print(
        f"SMOKE estimate: 1500 steps ~{wall / 25 * 1500 / 3600:.2f}h, "
        f"2000 steps ~{wall / 25 * 2000 / 3600:.2f}h, "
        f"2500 steps ~{wall / 25 * 2500 / 3600:.2f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
