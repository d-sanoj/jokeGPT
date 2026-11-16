"""API: health, corpus, experiments, train job lifecycle."""

import time


def test_health(client):
    """The service is up."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_corpus_stats(client):
    """Corpus stats match the cleaned corpus."""
    resp = client.get("/api/v1/corpus")
    assert resp.status_code == 200
    data = resp.json()
    assert data["jokes"] == 15000
    assert data["vocab_size"] == 95
    assert data["chars"] > 1_000_000


def test_generate_without_checkpoint_is_404(client):
    """Generating from a missing checkpoint is a clean 404."""
    resp = client.post(
        "/api/v1/generate",
        json={
            "prompt": "",
            "temperature": 0.8,
            "max_tokens": 20,
            "checkpoint": "no-such-run",
        },
    )
    assert resp.status_code == 404


def test_train_job_lifecycle(client):
    """A tiny training job runs to done and reports its loss."""
    resp = client.post(
        "/api/v1/train",
        json={
            "config": {
                "run_id": "api-test",
                "max_steps": 2,
                "eval_every": 2,
                "checkpoint_every": 100000,
                "data_subset": 200,
            }
        },
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    deadline = time.time() + 120
    while time.time() < deadline:
        status = client.get(f"/api/v1/train/{job_id}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(2)
    assert status["status"] == "done", status
    assert status["step"] == 2
    assert status["val_loss"] is not None


def test_train_unknown_job_is_404(client):
    """Polling an unknown job id is a clean 404."""
    resp = client.get("/api/v1/train/does-not-exist")
    assert resp.status_code == 404


def test_experiments_list(client):
    """The experiments endpoint returns a list (possibly empty)."""
    resp = client.get("/api/v1/experiments")
    assert resp.status_code == 200
    assert isinstance(resp.json()["runs"], list)
