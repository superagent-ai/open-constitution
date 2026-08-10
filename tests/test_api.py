from __future__ import annotations

from fastapi.testclient import TestClient

from api import modal_client
from api.main import app

client = TestClient(app)


def auth_headers(key: str = "platform-secret") -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_health_is_public():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_requires_configured_bearer_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert client.get("/v1/jobs/fc-test").status_code == 503

    monkeypatch.setenv("API_KEY", "platform-secret")
    assert client.get("/v1/jobs/fc-test", headers=auth_headers("wrong")).status_code == 401


def test_probe_training_spawns_fixed_job(monkeypatch):
    monkeypatch.setenv("API_KEY", "platform-secret")
    captured = {}

    async def fake_spawn(request, *, artifact_id):
        captured["request"] = request
        captured["artifact_id"] = artifact_id
        return "fc-probe"

    monkeypatch.setattr(modal_client, "spawn_probe", fake_spawn)
    response = client.post(
        "/v1/probes/train",
        headers=auth_headers(),
        json={"max_examples": 100, "epochs": 2},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "fc-probe"
    assert body["artifact_type"] == "probe"
    assert body["artifact_id"] == captured["artifact_id"]
    assert len(body["artifact_id"]) == 32
    assert captured["request"].max_examples == 100


def test_classifier_training_rejects_unapproved_model(monkeypatch):
    monkeypatch.setenv("API_KEY", "platform-secret")
    response = client.post(
        "/v1/classifiers/train",
        headers=auth_headers(),
        json={"model_id": "unapproved/model"},
    )

    assert response.status_code == 422


def test_publish_request_never_returns_hf_token(monkeypatch):
    monkeypatch.setenv("API_KEY", "platform-secret")
    captured = {}

    async def fake_spawn(request, *, artifact_id, artifact_type):
        captured["request_repr"] = repr(request)
        captured["artifact_id"] = artifact_id
        captured["artifact_type"] = artifact_type
        return "fc-publish"

    monkeypatch.setattr(modal_client, "spawn_publish", fake_spawn)
    response = client.post(
        f"/v1/artifacts/probe/{'a' * 32}/publish",
        headers=auth_headers(),
        json={
            "repo_id": "owner/probe",
            "hf_token": "hf_user_secret",
            "private": True,
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "job_id": "fc-publish",
        "status": "running",
    }
    assert "hf_user_secret" not in captured["request_repr"]


def test_cancel_job_returns_cleanup_result(monkeypatch):
    monkeypatch.setenv("API_KEY", "platform-secret")

    async def fake_cancel(job_id):
        return {
            "job_id": job_id,
            "artifact_id": "a" * 32,
            "status": "cancelled",
            "execution_cancelled": True,
            "artifacts_removed": True,
            "cleanup_errors": [],
        }

    monkeypatch.setattr(modal_client, "cancel_job", fake_cancel)
    response = client.post("/v1/jobs/fc-test/cancel", headers=auth_headers())

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["artifacts_removed"] is True


def test_cancel_unknown_job_returns_not_found(monkeypatch):
    monkeypatch.setenv("API_KEY", "platform-secret")

    async def fake_cancel(_job_id):
        return None

    monkeypatch.setattr(modal_client, "cancel_job", fake_cancel)
    response = client.post("/v1/jobs/fc-missing/cancel", headers=auth_headers())

    assert response.status_code == 404
