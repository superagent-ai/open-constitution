from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import HfApi


def validate_repo_id(repo_id: str) -> str:
    segment = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    if re.fullmatch(rf"{segment}/{segment}", repo_id) is None or ".." in repo_id or "--" in repo_id:
        raise ValueError("repo_id must use the form 'namespace/name'")
    return repo_id


def validate_artifact(artifact_dir: str | Path, artifact_type: str) -> Path:
    artifact_path = Path(artifact_dir)
    if not artifact_path.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_path}")

    if artifact_type == "probe":
        required = ["probe.pt", "config.json"]
    elif artifact_type == "classifier":
        required = ["config.json", "classifier_config.json"]
    else:
        raise ValueError("artifact_type must be 'probe' or 'classifier'")

    missing = [name for name in required if not (artifact_path / name).is_file()]
    if artifact_type == "classifier":
        if not any(
            (artifact_path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
        ):
            missing.append("model.safetensors or pytorch_model.bin")
        if not any(
            (artifact_path / name).is_file()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt")
        ):
            missing.append("tokenizer files")
    if missing:
        raise FileNotFoundError(f"Artifact is incomplete; missing: {', '.join(missing)}")
    return artifact_path


def write_probe_model_card(artifact_dir: str | Path, repo_id: str) -> None:
    artifact_path = Path(artifact_dir)
    config = json.loads((artifact_path / "config.json").read_text(encoding="utf-8"))
    model_id = config.get("model_id", "unknown")
    layer = config.get("layer", "unknown")
    content = f"""---
library_name: pytorch
tags:
  - activation-probe
  - safety
---

# {repo_id}

Linear safety activation probe trained for `{model_id}` at hidden-state layer `{layer}`.

This repository contains custom `probe.pt` weights and `config.json`. It is not a
standalone Transformers model; load it with the Open Constitution probe runtime.
"""
    (artifact_path / "README.md").write_text(content, encoding="utf-8")


def publish_artifact(
    *,
    artifact_dir: str | Path,
    artifact_id: str,
    artifact_type: str,
    repo_id: str,
    hf_token: str,
    private: bool = True,
    commit_message: str = "Upload trained Open Constitution artifact",
) -> dict[str, str]:
    artifact_path = validate_artifact(artifact_dir, artifact_type)
    validate_repo_id(repo_id)
    if not hf_token:
        raise ValueError("A Hugging Face token is required")

    if artifact_type == "probe":
        write_probe_model_card(artifact_path, repo_id)

    api = HfApi(token=hf_token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(artifact_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=["checkpoint-*", "checkpoint-*/**", "progress.json", "train.log"],
    )
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "repo_id": repo_id,
        "repo_url": str(repo_url),
    }
