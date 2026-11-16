"""Persist training checkpoints and experiment run records.

A checkpoint is weights only (no optimizer state): state_dict, model
config, vocab, step, and the loss history. That keeps each file small
(~4 MB for the main model) and makes generation reproducible.
"""

import json
from pathlib import Path

import torch

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CheckpointRepository:
    """Save/load model checkpoints and the runs.json record."""

    def __init__(self, checkpoints_dir=None, experiments_file=None):
        self.checkpoints_dir = checkpoints_dir or settings.checkpoints_dir
        self.experiments_file = experiments_file or settings.experiments_file

    def checkpoint_path(self, run_id: str, step: int) -> Path:
        """Path for one checkpoint file."""
        return self.checkpoints_dir / run_id / f"step-{step}.pt"

    def save_checkpoint(
        self,
        run_id: str,
        step: int,
        state_dict,
        config: dict,
        vocab: list[str],
        loss_history: list[dict],
    ) -> Path:
        """Write a checkpoint; returns its path."""
        path = self.checkpoint_path(run_id, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "step": step,
            "config": config,
            "vocab": vocab,
            "state_dict": {k: v.cpu() for k, v in state_dict.items()},
            "loss_history": loss_history,
        }
        torch.save(payload, path)
        logger.info(
            "saved checkpoint %s (%d bytes)", path, path.stat().st_size
        )
        return path

    def load_checkpoint(self, path: Path) -> dict:
        """Load a checkpoint file (weights only, CPU)."""
        return torch.load(path, map_location="cpu", weights_only=False)

    def list_checkpoints(self, run_id: str) -> list[Path]:
        """All checkpoints for a run, sorted by step."""
        folder = self.checkpoints_dir / run_id
        if not folder.exists():
            return []
        return sorted(
            folder.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1])
        )

    def latest_checkpoint(self, run_id: str) -> Path | None:
        """The highest-step checkpoint for a run, if any."""
        checkpoints = self.list_checkpoints(run_id)
        return checkpoints[-1] if checkpoints else None

    def save_runs(self, runs: list[dict]) -> None:
        """Write the experiment run records to runs.json."""
        self.experiments_file.parent.mkdir(parents=True, exist_ok=True)
        self.experiments_file.write_text(
            json.dumps({"runs": runs}, indent=2), encoding="utf-8"
        )

    def load_runs(self) -> list[dict]:
        """Read the experiment run records, or [] if none exist."""
        if not self.experiments_file.exists():
            return []
        return json.loads(
            self.experiments_file.read_text(encoding="utf-8")
        ).get("runs", [])
