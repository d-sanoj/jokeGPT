"""Training schemas."""

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Hyperparameters for one training run."""

    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    context_len: int = 128
    dropout: float = 0.0
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_steps: int = 1500
    eval_every: int = 100
    checkpoint_every: int = 500
    seed: int = 42
    data_subset: int | None = Field(
        default=None,
        description="Train on only the first N jokes (ablation).",
    )
    run_id: str = "main"


class TrainRequest(BaseModel):
    """Request to start a training job."""

    config: ModelConfig = Field(default_factory=ModelConfig)


class TrainJobStatus(BaseModel):
    """Status of a training job (polled by the client)."""

    job_id: str
    run_id: str
    status: str  # queued | running | done | failed
    step: int = 0
    max_steps: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    checkpoint: str | None = None


class LossPoint(BaseModel):
    """One logged (step, train loss, val loss) point."""

    step: int
    train_loss: float
    val_loss: float | None = None
