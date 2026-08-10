from __future__ import annotations

import os
from typing import Any

import modal
from modal.exception import OutputExpiredError

from .schemas import ClassifierTrainRequest, ProbeTrainRequest, PublishRequest

DEFAULT_MODAL_APP_NAME = "open-constitution"
PROBE_DATA_PATH = "data/training_data.jsonl"
CLASSIFIER_DATA_PATH = "data/training_data_classifier.jsonl"


def _function(name: str) -> modal.Function:
    return modal.Function.from_name(
        os.environ.get("MODAL_APP_NAME", DEFAULT_MODAL_APP_NAME),
        name,
        environment_name=os.environ.get("MODAL_ENVIRONMENT"),
    )


async def spawn_probe(
    request: ProbeTrainRequest,
    *,
    artifact_id: str,
) -> str:
    call = await _function("train_probe").spawn.aio(
        model_id=request.model_id,
        data_path=PROBE_DATA_PATH,
        layer=request.layer,
        out_dir=f"jobs/{artifact_id}/probe",
        epochs=request.epochs,
        lr=request.learning_rate,
        max_examples=request.max_examples,
        no_chat_template=request.no_chat_template,
        artifact_id=artifact_id,
    )
    return call.object_id


async def spawn_classifier(
    request: ClassifierTrainRequest,
    *,
    artifact_id: str,
) -> str:
    call = await _function("run_train_exchange_classifier").spawn.aio(
        model_id=request.model_id,
        data_path=CLASSIFIER_DATA_PATH,
        out_dir=f"jobs/{artifact_id}/classifier",
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        max_length=request.max_length,
        prefix_augment=request.prefix_augment,
        prefix_copies=request.prefix_copies,
        logging_steps=request.logging_steps,
        save_steps=request.save_steps,
        artifact_id=artifact_id,
    )
    return call.object_id


async def spawn_publish(
    request: PublishRequest,
    *,
    artifact_id: str,
    artifact_type: str,
) -> str:
    secret = modal.Secret.from_dict({"HF_TOKEN": request.hf_token.get_secret_value()})
    publisher = _function("publish_artifact_to_hf").with_options(secrets=[secret])
    call = await publisher.spawn.aio(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        repo_id=request.repo_id,
        private=request.private,
        commit_message=request.commit_message,
    )
    return call.object_id


async def poll_job(job_id: str) -> dict[str, Any]:
    call = modal.FunctionCall.from_id(job_id)
    try:
        result = await call.get.aio(timeout=0)
    except TimeoutError:
        return {"job_id": job_id, "status": "running", "result": None}
    except OutputExpiredError:
        return {"job_id": job_id, "status": "expired", "result": None}
    except Exception as exc:
        return {
            "job_id": job_id,
            "status": "failed",
            "result": None,
            "error": {
                "type": type(exc).__name__,
                "message": "The Modal job failed; inspect Modal logs for details.",
            },
        }
    return {"job_id": job_id, "status": "succeeded", "result": result}
