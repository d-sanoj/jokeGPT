"""Generation endpoint: sample jokes from a trained checkpoint."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.deps import generation_service
from app.models.generate import GenerateRequest, GenerateResponse
from app.services.generation_service import GenerationService

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("", response_model=GenerateResponse)
def generate_text(
    request: GenerateRequest,
    service: GenerationService = Depends(generation_service),
):
    """Sample one joke from a checkpoint (temperature + top-k)."""
    try:
        return service.generate(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            checkpoint=request.checkpoint,
            seed=request.seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
