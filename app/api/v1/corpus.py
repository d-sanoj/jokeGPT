"""Corpus endpoints."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import corpus_repository
from app.models.corpus import CorpusStats
from app.repositories.corpus_repository import CorpusRepository

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("", response_model=CorpusStats)
def get_corpus_stats(
    repository: CorpusRepository = Depends(corpus_repository),
):
    """Corpus summary: joke count, characters, vocabulary, source."""
    return repository.stats()
