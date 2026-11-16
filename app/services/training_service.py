"""Training loop and background training jobs.

The synchronous :meth:`TrainingService.train` is the core used by both
the experiment script and the API background job. Training runs on CPU
(no GPU on this machine), so every run records wall-clock time.
"""

import math
import threading
import time
import uuid

import numpy as np
import torch

from app.core.logging import get_logger
from app.models.train import LossPoint, ModelConfig, TrainJobStatus
from app.repositories.checkpoint_repository import CheckpointRepository
from app.repositories.corpus_repository import CorpusRepository
from app.services.model_service import (
    build_model,
    count_parameters,
    model_config_dict,
)
from app.services.tokenizer_service import CharTokenizer

logger = get_logger(__name__)

TERMINAL_PUNCT = (".", "!", "?")


class TrainingService:
    """Owns the tokenizer, the model, and the training jobs."""

    def __init__(
        self,
        corpus_repository: CorpusRepository,
        checkpoint_repository: CheckpointRepository,
    ):
        self.corpus = corpus_repository
        self.checkpoints = checkpoint_repository
        self._tokenizer: CharTokenizer | None = None
        self._jobs: dict[str, TrainJobStatus] = {}
        self._lock = threading.Lock()

    def tokenizer(self) -> CharTokenizer:
        """The corpus tokenizer (built once, cached)."""
        if self._tokenizer is None:
            jokes = self.corpus.load_jokes()
            self._tokenizer = CharTokenizer.from_texts([j.text for j in jokes])
        return self._tokenizer

    def _encode_split(self, jokes, tokenizer: CharTokenizer):
        """Join jokes with newlines and encode to a token tensor."""
        text = "\n".join(j.text for j in jokes) + "\n"
        ids = tokenizer.encode(text)
        return torch.tensor(ids, dtype=torch.long)

    def _batch(self, ids: torch.Tensor, batch_size: int, context_len: int):
        """Random windows of (input, next-token target)."""
        max_start = len(ids) - context_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([ids[s : s + context_len] for s in starts])
        y = torch.stack([ids[s + 1 : s + context_len + 1] for s in starts])
        return x, y

    def _val_loss(
        self, model, val_ids, config: ModelConfig, batches: int = 32
    ) -> float:
        """Mean loss on fixed seeded val windows."""
        model.eval()
        gen = torch.Generator().manual_seed(config.seed + 1)
        losses = []
        with torch.no_grad():
            for _ in range(batches):
                max_start = len(val_ids) - config.context_len - 1
                starts = torch.randint(
                    0, max_start, (config.batch_size,), generator=gen
                )
                x = torch.stack(
                    [val_ids[s : s + config.context_len] for s in starts]
                )
                y = torch.stack(
                    [
                        val_ids[s + 1 : s + config.context_len + 1]
                        for s in starts
                    ]
                )
                _, loss = model(x, y)
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    def _lr_at(self, step: int, config: ModelConfig) -> float:
        """Warmup then cosine decay to 10% of the peak learning rate."""
        if step < config.warmup_steps:
            return config.learning_rate * step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return config.learning_rate * (0.1 + 0.9 * cosine)

    def train(self, config: ModelConfig, progress=None) -> dict:
        """Run one training job synchronously; returns its record.

        ``progress`` is an optional callback(job_status) invoked after
        each eval so the API can report live step and loss.
        """
        torch.manual_seed(config.seed)
        t0 = time.time()
        tokenizer = self.tokenizer()
        jokes = self.corpus.load_jokes()
        if config.data_subset is not None:
            jokes = jokes[: config.data_subset]
        train_jokes, val_jokes = self._split(jokes, config)
        train_ids = self._encode_split(train_jokes, tokenizer)
        val_ids = self._encode_split(val_jokes, tokenizer)
        model_cfg = model_config_dict(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            n_layer=config.n_layer,
            n_head=config.n_head,
            context_len=config.context_len,
            dropout=config.dropout,
        )
        model = build_model(model_cfg)
        params = count_parameters(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        logger.info(
            "training %s: %d params, %d train tokens, %d steps",
            config.run_id,
            params,
            len(train_ids),
            config.max_steps,
        )
        loss_history: list[dict] = []
        checkpoints: list[str] = []
        train_losses: list[float] = []
        for step in range(1, config.max_steps + 1):
            for group in optimizer.param_groups:
                group["lr"] = self._lr_at(step, config)
            x, y = self._batch(
                train_ids, config.batch_size, config.context_len
            )
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            if step % config.eval_every == 0 or step == config.max_steps:
                val = self._val_loss(model, val_ids, config)
                point = LossPoint(
                    step=step,
                    train_loss=float(np.mean(train_losses[-50:])),
                    val_loss=val,
                )
                loss_history.append(point.model_dump())
                logger.info(
                    "%s step %d/%d train %.3f val %.3f",
                    config.run_id,
                    step,
                    config.max_steps,
                    point.train_loss,
                    val,
                )
                if progress is not None:
                    progress(step, point.train_loss, val)
            if step % config.checkpoint_every == 0 or step == config.max_steps:
                path = self.checkpoints.save_checkpoint(
                    config.run_id,
                    step,
                    model.state_dict(),
                    model_cfg,
                    tokenizer.vocab,
                    loss_history,
                )
                checkpoints.append(str(path))
        wall = time.time() - t0
        final_train = float(np.mean(train_losses[-50:]))
        final_val = loss_history[-1]["val_loss"] if loss_history else None
        record = {
            "run_id": config.run_id,
            "label": config.run_id,
            "config": config.model_dump(),
            "train_loss": final_train,
            "val_loss": final_val,
            "wall_seconds": round(wall, 1),
            "steps": config.max_steps,
            "params": params,
            "samples": [],
            "loss_history": loss_history,
            "quality": None,
            "checkpoints": checkpoints,
        }
        logger.info(
            "%s done: val %.3f in %.0fs", config.run_id, final_val or -1, wall
        )
        return record

    @staticmethod
    def _split(jokes, config: ModelConfig):
        """Seeded train/val split of a joke list."""
        import random

        items = list(jokes)
        random.Random(config.seed).shuffle(items)
        cut = int(len(items) * 0.95)
        return items[:cut], items[cut:]

    # -- background jobs (API) -------------------------------------------

    def start_job(self, config: ModelConfig) -> TrainJobStatus:
        """Create a job record and run it in a background thread."""
        job_id = uuid.uuid4().hex[:12]
        status = TrainJobStatus(
            job_id=job_id,
            run_id=config.run_id,
            status="queued",
            max_steps=config.max_steps,
        )
        with self._lock:
            self._jobs[job_id] = status
        thread = threading.Thread(
            target=self._run_job, args=(job_id, config), daemon=True
        )
        thread.start()
        return status

    def _run_job(self, job_id: str, config: ModelConfig) -> None:
        """Background worker: train and update the job record."""
        t0 = time.time()
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
        try:

            def progress(step, train_loss, val_loss):
                with self._lock:
                    job = self._jobs[job_id]
                    job.step = step
                    job.train_loss = train_loss
                    job.val_loss = val_loss
                    job.elapsed_seconds = time.time() - t0

            record = self.train(config, progress=progress)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "done"
                job.step = config.max_steps
                job.train_loss = record["train_loss"]
                job.val_loss = record["val_loss"]
                job.elapsed_seconds = time.time() - t0
                if record["checkpoints"]:
                    job.checkpoint = record["checkpoints"][-1]
        except Exception as exc:  # noqa: BLE001 - report to the client
            logger.exception("training job %s failed", job_id)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.elapsed_seconds = time.time() - t0

    def get_job(self, job_id: str) -> TrainJobStatus | None:
        """Current status of a training job, if it exists."""
        with self._lock:
            return self._jobs.get(job_id)
