"""Application settings.

All paths default to the bundled ``data/`` and ``checkpoints/`` directories,
so the service is self-contained: it never reads the sine-portfolio checkout
at runtime. Every setting can be overridden with an environment variable
prefixed ``JOKEGPT_`` (e.g. ``JOKEGPT_PORT=8001``).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the jokeGPT service."""

    model_config = SettingsConfigDict(env_prefix="JOKEGPT_")

    jokes_file: Path = BASE_DIR / "data" / "jokes.jsonl"
    checkpoints_dir: Path = BASE_DIR / "checkpoints"
    visuals_dir: Path = BASE_DIR / "visuals"
    experiments_file: Path = BASE_DIR / "experiments" / "runs.json"

    # Main model defaults (the config trained and shipped as "main").
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    context_len: int = 128
    dropout: float = 0.0
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    train_split: float = 0.95
    seed: int = 42

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
