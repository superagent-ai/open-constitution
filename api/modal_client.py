from __future__ import annotations

import os
import re
from typing import Any

import modal
from modal.exception import InvalidError, OutputExpiredError

from activation_probe_mvp.probe_models import resolve_probe_model_route

from .schemas import ClassifierTrainRequest, ProbeTrainRequest, PublishRequest

DEFAULT_MODAL_APP_NAME = "open-constitution"
PROGRESS_DICT_NAME = "open-constitution-job-progress"
OUTPUT_VOLUME_NAME = "open-constitution-outputs"
PROBE_DATA_PATH = "data/training_data.jsonl"
CLASSIFIER_DATA_PATH = "data/training_data_classifier.jsonl"


def _function(name: str) -> modal.Function:
    return modal.Function.from_name(
        os.environ.get("MODAL_APP_NAME", DEFAULT_MODAL_APP_NAME),
        name,
        environment_name=os.environ.get("MODAL_ENVIRONMENT"),
    )


def _progress_store() -> modal.Dict:
    return modal.Dict.from_name(
        PROGRESS_DICT_NAME,
        environment_name=os.environ.get("MODAL_ENVIRONMENT"),
        create_if_missing=True,
    )


async def _register_job(
    job_id: str,
    artifact_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        store = _progress_store()
        await store.put.aio(f"job:{job_id}", artifact_id)
        await store.put.aio(
            f"progress:{artifact_id}",
            {
                "artifact_id": artifact_id,
                "phase": "queued",
                "percent": 0.0,
                **(metadata or {}),
            },
            skip_if_exists=True,
        )
    except Exception:
        return


async def _job_progress(job_id: str) -> dict[str, Any] | None:
    try:
        store = _progress_store()
        artifact_id = await store.get.aio(f"job:{job_id}")
        if artifact_id is None:
            return None
        return await store.get.aio(f"progress:{artifact_id}")
    except Exception:
        return None


async def cancel_job(job_id: str) -> dict[str, Any] | None:
    store = _progress_store()
    artifact_id = await store.get.aio(f"job:{job_id}")
    if artifact_id is None or re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None:
        return None

    call = modal.FunctionCall.from_id(job_id)
    await call.cancel.aio()

    cleanup_errors: list[str] = []
    volume = modal.Volume.from_name(
        OUTPUT_VOLUME_NAME,
        environment_name=os.environ.get("MODAL_ENVIRONMENT"),
    )
    try:
        await volume.remove_file.aio(f"jobs/{artifact_id}", recursive=True)
    except FileNotFoundError:
        pass
    except InvalidError as exc:
        if "No such file or directory" not in str(exc):
            cleanup_errors.append(type(exc).__name__)
    except Exception as exc:
        cleanup_errors.append(type(exc).__name__)

    if not cleanup_errors:
        try:
            await store.pop.aio(f"progress:{artifact_id}", None)
            await store.pop.aio(f"job:{job_id}", None)
        except Exception as exc:
            cleanup_errors.append(type(exc).__name__)

    return {
        "job_id": job_id,
        "artifact_id": artifact_id,
        "status": "cancelled",
        "execution_cancelled": True,
        "artifacts_removed": not cleanup_errors,
        "cleanup_errors": cleanup_errors,
    }


async def spawn_probe(
    request: ProbeTrainRequest,
    *,
    artifact_id: str,
) -> str:
    route = resolve_probe_model_route(request.model_id)
    worker = _function("train_probe").with_options(
        gpu=route.gpu,
        memory=route.memory_mib,
    )
    call = await worker.spawn.aio(
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
    await _register_job(
        call.object_id,
        artifact_id,
        metadata={
            "model_id": route.model_id,
            "model_family": route.family,
            "gpu": route.gpu,
            "memory_mib": route.memory_mib,
            "estimated_parameters_b": route.estimated_parameters_b,
        },
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
    await _register_job(call.object_id, artifact_id)
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
        return {
            "job_id": job_id,
            "status": "running",
            "result": None,
            "progress": await _job_progress(job_id),
        }
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
