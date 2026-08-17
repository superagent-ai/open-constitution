from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

from activation_probe_mvp.probe_models import (
    DEFAULT_PROBE_MODEL_ID,
    ProbeFamily,
    validate_probe_model_id,
)

ArtifactType = Literal["probe", "classifier"]

PROBE_MODEL_ID = DEFAULT_PROBE_MODEL_ID
CLASSIFIER_MODEL_ID = "answerdotai/ModernBERT-base"


class ProbeTrainRequest(BaseModel):
    model_id: str = PROBE_MODEL_ID
    layer: int = Field(default=-4, ge=-128, le=127)
    epochs: int = Field(default=100, ge=1, le=500)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    max_examples: int = Field(default=20000, ge=4, le=20000)
    no_chat_template: bool = False

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return validate_probe_model_id(value)


class ClassifierTrainRequest(BaseModel):
    model_id: Literal["answerdotai/ModernBERT-base"] = CLASSIFIER_MODEL_ID
    epochs: float = Field(default=5, gt=0, le=50)
    batch_size: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=2e-5, gt=0, le=1)
    max_length: int = Field(default=512, ge=32, le=4096)
    prefix_augment: bool = True
    prefix_copies: int = Field(default=1, ge=1, le=10)
    logging_steps: int = Field(default=500, ge=1, le=100000)
    save_steps: int = Field(default=5000, ge=1, le=100000)


class PublishRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=192)
    hf_token: SecretStr
    private: bool = True
    commit_message: str = Field(
        default="Upload trained Open Constitution artifact",
        min_length=1,
        max_length=200,
    )

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        segment = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
        if re.fullmatch(rf"{segment}/{segment}", value) is None or ".." in value or "--" in value:
            raise ValueError("repo_id must use the form 'namespace/name'")
        return value

    @field_validator("hf_token")
    @classmethod
    def validate_hf_token(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("hf_token must not be empty")
        return value


class SpawnedJob(BaseModel):
    job_id: str
    status: Literal["running"] = "running"
    artifact_id: str | None = None
    artifact_type: ArtifactType | None = None
    model_id: str | None = None
    model_family: ProbeFamily | None = None
    gpu: str | None = None
    memory_mib: int | None = None
