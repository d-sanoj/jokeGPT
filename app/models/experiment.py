"""Experiment schemas."""

from pydantic import BaseModel


class ExperimentRun(BaseModel):
    """One recorded training run: config, metrics, samples."""

    run_id: str
    label: str
    config: dict
    train_loss: float | None = None
    val_loss: float | None = None
    wall_seconds: float = 0.0
    steps: int = 0
    params: int = 0
    samples: list[str] = []
    loss_history: list[dict] = []
    quality: dict | None = None


class ExperimentSummary(BaseModel):
    """All recorded runs, newest first."""

    runs: list[ExperimentRun]
