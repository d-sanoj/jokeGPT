"""Generation schemas."""

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request to generate text from a trained checkpoint."""

    prompt: str = ""
    temperature: float = 0.8
    max_tokens: int = 120
    top_k: int = 40
    checkpoint: str = "main"
    seed: int | None = None


class GenerateResponse(BaseModel):
    """Generated text plus the checkpoint it came from."""

    text: str
    checkpoint: str
    temperature: float
    tokens_generated: int


class QualitySignals(BaseModel):
    """Automated quality signals for a batch of generated jokes."""

    n_samples: int
    complete_pct: float
    duplicate_pct: float
    mean_chars: float
    mean_words: float
