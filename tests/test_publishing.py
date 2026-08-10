from __future__ import annotations

import json

import pytest

from activation_probe_mvp import publishing


class FakeHfApi:
    instances: list["FakeHfApi"] = []

    def __init__(self, *, token: str):
        self.token = token
        self.created: dict | None = None
        self.uploaded: dict | None = None
        self.__class__.instances.append(self)

    def create_repo(self, **kwargs):
        self.created = kwargs
        return f"https://huggingface.co/{kwargs['repo_id']}"

    def upload_folder(self, **kwargs):
        self.uploaded = kwargs


def test_publish_probe_creates_model_card_and_uploads(monkeypatch, tmp_path):
    monkeypatch.setattr(publishing, "HfApi", FakeHfApi)
    FakeHfApi.instances.clear()
    (tmp_path / "probe.pt").write_bytes(b"weights")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_id": "example/base", "layer": -4}),
        encoding="utf-8",
    )

    result = publishing.publish_artifact(
        artifact_dir=tmp_path,
        artifact_id="a" * 32,
        artifact_type="probe",
        repo_id="example/probe",
        hf_token="hf_secret",
    )

    api = FakeHfApi.instances[0]
    assert api.created == {
        "repo_id": "example/probe",
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }
    assert api.uploaded is not None
    assert api.uploaded["folder_path"] == str(tmp_path)
    assert "example/base" in (tmp_path / "README.md").read_text(encoding="utf-8")
    assert result["repo_url"] == "https://huggingface.co/example/probe"
    assert "hf_secret" not in repr(result)


def test_validate_classifier_requires_weights_and_tokenizer(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "classifier_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="model.safetensors or pytorch_model.bin"):
        publishing.validate_artifact(tmp_path, "classifier")


@pytest.mark.parametrize("repo_id", ["missing-namespace", "../escape", "owner/name/extra"])
def test_validate_repo_id_rejects_invalid_values(repo_id):
    with pytest.raises(ValueError, match="namespace/name"):
        publishing.validate_repo_id(repo_id)
