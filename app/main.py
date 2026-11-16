"""FastAPI application factory.

The app is built by :func:`create_app`, which wires the v1 routers and
creates the service singletons (repositories, training, generation) once
at startup so the corpus is loaded and checkpoints are cached across
requests. Run it with ``python -m app`` or ``uvicorn app.main:app``.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.v1 import router as v1_router
from app.api.v1.health import router as health_router
from app.core.logging import configure_logging, get_logger
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.generation_service import GenerationService
from app.services.training_service import TrainingService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load the corpus once. Shutdown: nothing to clean up."""
    configure_logging()
    corpus_repository = CorpusRepository()
    checkpoint_repository = CheckpointRepository()
    stats = corpus_repository.stats()
    app.state.corpus_repository = corpus_repository
    app.state.checkpoint_repository = checkpoint_repository
    app.state.training_service = TrainingService(
        corpus_repository, checkpoint_repository
    )
    app.state.generation_service = GenerationService(checkpoint_repository)
    logger.info(
        "corpus loaded: %d jokes, %d chars, vocab %d",
        stats.jokes,
        stats.chars,
        stats.vocab_size,
    )
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="jokeGPT",
        description=(
            "A tiny GPT that tells jokes, as an API service. "
            "Trained from scratch on short jokes: character-level "
            "transformer, ~1M parameters, CPU only."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(v1_router, prefix="/api/v1")
    return app


app = create_app()
