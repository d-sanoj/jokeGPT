"""Health check."""

from fastapi import APIRouter

from app import __version__

router = APIRouter()


@router.get("/health")
def health():
    """Liveness probe: the service is up."""
    return {"status": "ok", "version": __version__}
