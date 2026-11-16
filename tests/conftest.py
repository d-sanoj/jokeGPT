"""Shared fixtures: repository, services, and the API test client."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.generation_service import GenerationService
from app.services.training_service import TrainingService


@pytest.fixture(scope="session")
def repository():
    """The corpus repository (loads the bundled jokes once)."""
    return CorpusRepository()


@pytest.fixture(scope="session")
def checkpoint_repository():
    """The checkpoint repository."""
    return CheckpointRepository()


@pytest.fixture(scope="session")
def training_service(repository, checkpoint_repository):
    """The training service."""
    return TrainingService(repository, checkpoint_repository)


@pytest.fixture(scope="session")
def generation_service(checkpoint_repository):
    """The generation service."""
    return GenerationService(checkpoint_repository)


@pytest.fixture(scope="session")
def client():
    """A TestClient for the API (lifespan loads the corpus)."""
    with TestClient(create_app()) as test_client:
        yield test_client
