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


def test_generate_from_trained_word_level_model():
    """The trained word-level model generates non-empty text."""
    repo = CheckpointRepository()
    path = repo.latest_checkpoint("word-level")
    if path is None:
        pytest.skip("no trained word-level checkpoint yet")
    service = GenerationService(repo)
    resp = service.generate(
        temperature=0.8, checkpoint="word-level", seed=7, max_tokens=120
    )
    assert resp.text
    assert len(resp.text) >= 10
    assert resp.checkpoint == "word-level"
    assert resp.tokens_generated > 0


def test_generate_from_trained_big_model():
    """The trained step-3 big model (the current best) generates text."""
    repo = CheckpointRepository()
    path = repo.latest_checkpoint("word-level-50k-big")
    if path is None:
        pytest.skip("no trained word-level-50k-big checkpoint yet")
    service = GenerationService(repo)
    resp = service.generate(
        temperature=0.8,
        checkpoint="word-level-50k-big",
        seed=7,
        max_tokens=120,
    )
    assert resp.text
    assert len(resp.text) >= 10
    assert resp.checkpoint == "word-level-50k-big"
    assert resp.tokens_generated > 0


def test_top_p_nucleus_filtering():
    """top_p=1.0 is a no-op; a tiny top_p keeps only the top token."""
    service = GenerationService(CheckpointRepository())
    import torch

    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    kept = service._nucleus_filter(logits, 1.0)
    assert torch.equal(kept, logits)
    tiny = service._nucleus_filter(logits, 0.01)
    assert tiny[0] == 3.0
    assert torch.isinf(tiny[1]) and tiny[1] < 0
