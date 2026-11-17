"""Generation schemas."""

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request to generate text from a trained checkpoint."""

    prompt: str = ""
    temperature: float = 0.8
    max_tokens: int = 120
    top_k: int = 40
    top_p: float = 1.0
    # Default is the best checkpoint so far: step 3, word-level-50k-big
    # (d_model 256, 6 layers, 40k vocab, 50k jokes; val loss 5.225,
    # 70% complete, 0% good). Updated each step as a better one lands.
    checkpoint: str = "word-level-50k-big"
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
