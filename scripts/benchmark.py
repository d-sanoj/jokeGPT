#!/usr/bin/env python3
"""Measure training speed: run 30 steps of the main config and report
seconds per step, so the experiment step counts can be sized to fit
the compute budget. Usage: python scripts/benchmark.py
"""

import time

from app.models.train import ModelConfig
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.training_service import TrainingService


def main():
    service = TrainingService(CorpusRepository(), CheckpointRepository())
    config = ModelConfig(
        run_id="bench", max_steps=30, eval_every=15, checkpoint_every=100000
    )
    t0 = time.time()
    record = service.train(config)
    wall = time.time() - t0
    print(f"params:        {record['params']:,}")
    print(f"steps:         {record['steps']}")
    print(f"wall:          {wall:.1f}s")
    print(f"sec/step:      {wall / record['steps']:.2f}")
    print(f"train loss:    {record['train_loss']:.3f}")
    print(f"val loss:      {record['val_loss']:.3f}")
    print(
        f"projected: 1500 steps = {wall / record['steps'] * 1500 / 60:.0f} min"
    )


if __name__ == "__main__":
    main()
