"""Training resume: continue from a checkpoint without redoing work."""

import tempfile
from pathlib import Path

from app.models.train import ModelConfig
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.training_service import TrainingService


def _tiny_config(run_id: str, max_steps: int) -> ModelConfig:
    """A tiny, fast config for the resume test."""
    return ModelConfig(
        run_id=run_id,
        d_model=32,
        n_layer=1,
        n_head=2,
        context_len=16,
        batch_size=8,
        max_steps=max_steps,
        eval_every=2,
        checkpoint_every=2,
        data_subset=50,
    )


def test_resume_continues_from_checkpoint():
    """Resuming keeps the loss history and reaches the final step."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = CheckpointRepository(checkpoints_dir=Path(tmp) / "ckpt")
        service = TrainingService(CorpusRepository(), ckpt)
        first = service.train(_tiny_config("resume-test", max_steps=4))
        assert first["loss_history"][-1]["step"] == 4
        assert first["resumed_from_step"] is None
        # A second run to the same max_steps now skips (final exists).
        # Extend the target instead: resume to step 6 from step 4.
        resumed = service.train(
            _tiny_config("resume-test", max_steps=6),
            resume_from=ckpt.checkpoint_path("resume-test", 4),
        )
        assert resumed["resumed_from_step"] == 4
        steps = [h["step"] for h in resumed["loss_history"]]
        assert steps == [2, 4, 6]
        assert ckpt.checkpoint_path("resume-test", 6).exists()
