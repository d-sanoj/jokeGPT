"""Generation smoke test: load the trained main model and generate.

Skipped until the experiment suite has produced checkpoints/main.
"""

import pytest

from app.repositories.checkpoint_repository import CheckpointRepository
from app.services.generation_service import GenerationService


def test_generate_from_trained_main_model():
    """The trained main model generates non-empty text."""
    repo = CheckpointRepository()
    path = repo.latest_checkpoint("main")
    if path is None:
        pytest.skip("no trained main checkpoint yet")
    service = GenerationService(repo)
    resp = service.generate(
        temperature=0.8, checkpoint="main", seed=7, max_tokens=120
    )
    assert resp.text
    assert len(resp.text) >= 10
    assert resp.checkpoint == "main"
    assert resp.tokens_generated > 0
