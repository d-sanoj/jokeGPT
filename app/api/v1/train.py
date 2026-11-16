"""Training endpoints: start a job, poll its status."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.deps import training_service
from app.models.train import TrainJobStatus, TrainRequest
from app.services.training_service import TrainingService

router = APIRouter(prefix="/train", tags=["train"])


@router.post("", response_model=TrainJobStatus, status_code=202)
def start_training(
    request: TrainRequest,
    service: TrainingService = Depends(training_service),
):
    """Start a training job in the background; returns its job id."""
    return service.start_job(request.config)


@router.get("/{job_id}", response_model=TrainJobStatus)
def get_training_job(
    job_id: str,
    service: TrainingService = Depends(training_service),
):
    """Current status of a training job: step, loss, elapsed time."""
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
