"""Experiment records: the runs from the experiment suite."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.deps import checkpoint_repository
from app.models.experiment import ExperimentRun, ExperimentSummary
from app.repositories.checkpoint_repository import CheckpointRepository

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=ExperimentSummary)
def list_experiments(
    repository: CheckpointRepository = Depends(checkpoint_repository),
):
    """All recorded experiment runs, newest first."""
    runs = [ExperimentRun(**r) for r in repository.load_runs()]
    return ExperimentSummary(runs=runs)


@router.get("/{run_id}", response_model=ExperimentRun)
def get_experiment(
    run_id: str,
    repository: CheckpointRepository = Depends(checkpoint_repository),
):
    """One recorded run: config, metrics, samples, loss history."""
    for record in repository.load_runs():
        if record.get("run_id") == run_id:
            return ExperimentRun(**record)
    raise HTTPException(status_code=404, detail="run not found")
