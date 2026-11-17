"""Text generation from a trained checkpoint.

Sampling is temperature + top-k, stopping at the first newline (the end
of a joke) or ``max_tokens``. Generation is deterministic when a seed
is given, so samples are reproducible.
"""

import torch

from app.core.logging import get_logger
from app.models.generate import GenerateResponse
from app.repositories.checkpoint_repository import CheckpointRepository
from app.services.model_service import build_model
from app.services.tokenizer_service import CharTokenizer, WordTokenizer

logger = get_logger(__name__)


class GenerationService:
    """Loads checkpoints (cached) and samples text from them."""

    def __init__(self, checkpoint_repository: CheckpointRepository):
        self.checkpoints = checkpoint_repository
        self._models: dict[str, tuple] = {}

    def _load(self, run_id: str, step: int | None = None):
        """Load (model, tokenizer) for a run, cached by path."""
        if step is None:
            path = self.checkpoints.latest_checkpoint(run_id)
            if path is None:
                raise FileNotFoundError(
                    f"no checkpoint found for run {run_id!r}"
                )
        else:
            path = self.checkpoints.checkpoint_path(run_id, step)
            if not path.exists():
                raise FileNotFoundError(f"checkpoint not found: {path}")
        key = str(path)
        if key not in self._models:
            payload = self.checkpoints.load_checkpoint(path)
            if payload.get("tokenizer", "char") == "word":
                tokenizer = WordTokenizer(payload["vocab"])
            else:
                tokenizer = CharTokenizer(payload["vocab"])
            model = build_model(payload["config"])
            model.load_state_dict(payload["state_dict"])
            model.eval()
            self._models[key] = (model, tokenizer, payload)
            logger.info("loaded checkpoint %s", path)
        return self._models[key]

    def generate(
        self,
        prompt: str = "",
        temperature: float = 0.8,
        max_tokens: int = 120,
        top_k: int = 40,
        top_p: float = 1.0,
        checkpoint: str = "word-level-50k-big",
        seed: int | None = None,
        stop_at_eos: bool = True,
    ) -> GenerateResponse:
        """Sample one joke from a checkpoint."""
        model, tokenizer, _ = self._load(checkpoint)
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [tokenizer.start_token_id]
        generated = 0
        with torch.no_grad():
            while generated < max_tokens:
                ctx = torch.tensor([ids], dtype=torch.long)
                logits = model.generate_step(ctx)
                logits = logits[0] / max(temperature, 1e-6)
                if top_k > 0:
                    top_values, _ = torch.topk(logits, top_k)
                    logits[logits < top_values[-1]] = -float("inf")
                if top_p < 1.0:
                    logits = self._nucleus_filter(logits, top_p)
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=gen))
                ids.append(nxt)
                generated += 1
                if (
                    stop_at_eos
                    and tokenizer.stop_token_id is not None
                    and nxt == tokenizer.stop_token_id
                ):
                    break
        text = tokenizer.decode(ids).strip()
        return GenerateResponse(
            text=text,
            checkpoint=checkpoint,
            temperature=temperature,
            tokens_generated=generated,
        )

    @staticmethod
    def _nucleus_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Keep the smallest token set with cumulative prob >= top_p."""
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum - sorted_probs > top_p
        sorted_logits[remove] = -float("inf")
        return torch.full_like(logits, -float("inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    def generate_batch(
        self,
        n: int,
        temperature: float = 0.8,
        checkpoint: str = "word-level-50k-big",
        seed: int = 42,
        max_tokens: int = 120,
        top_k: int = 40,
        top_p: float = 1.0,
    ) -> list[str]:
        """Sample n jokes (seeded) for quality measurement."""
        return [
            self.generate(
                temperature=temperature,
                checkpoint=checkpoint,
                seed=seed + i,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
            ).text
            for i in range(n)
        ]
