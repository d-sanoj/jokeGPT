"""Request-scoped access to the services built at startup."""

from fastapi import Request

from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.generation_service import GenerationService
from app.services.training_service import TrainingService


def corpus_repository(request: Request) -> CorpusRepository:
    """The corpus repository built at startup."""
    return request.app.state.corpus_repository


def checkpoint_repository(request: Request) -> CheckpointRepository:
    """The checkpoint repository built at startup."""
    return request.app.state.checkpoint_repository


def training_service(request: Request) -> TrainingService:
    """The training service built at startup."""
    return request.app.state.training_service


def generation_service(request: Request) -> GenerationService:
    """The generation service built at startup."""
    return request.app.state.generation_service
