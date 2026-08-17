from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, status

from activation_probe_mvp.probe_models import resolve_probe_model_route

from . import modal_client
from .auth import require_api_key
from .schemas import (
    ArtifactType,
    ClassifierTrainRequest,
    ProbeTrainRequest,
    PublishRequest,
    SpawnedJob,
)

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


def _spawn_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Modal rejected the job submission",
    )


@router.post(
    "/probes/train",
    response_model=SpawnedJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def train_probe(request: ProbeTrainRequest) -> SpawnedJob:
    artifact_id = uuid4().hex
    route = resolve_probe_model_route(request.model_id)
    try:
        job_id = await modal_client.spawn_probe(request, artifact_id=artifact_id)
    except Exception as exc:
        raise _spawn_error() from exc
    return SpawnedJob(
        job_id=job_id,
        artifact_id=artifact_id,
        artifact_type="probe",
        model_id=route.model_id,
        model_family=route.family,
        gpu=route.gpu,
        memory_mib=route.memory_mib,
    )


@router.post(
    "/classifiers/train",
    response_model=SpawnedJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def train_classifier(request: ClassifierTrainRequest) -> SpawnedJob:
    artifact_id = uuid4().hex
    try:
        job_id = await modal_client.spawn_classifier(request, artifact_id=artifact_id)
    except Exception as exc:
        raise _spawn_error() from exc
    return SpawnedJob(
        job_id=job_id,
        artifact_id=artifact_id,
        artifact_type="classifier",
    )


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str = Path(min_length=3, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
):
    return await modal_client.poll_job(job_id)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str = Path(min_length=3, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
):
    try:
        result = await modal_client.cancel_job(job_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Modal rejected the cancellation request",
        ) from exc
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return result


@router.post(
    "/artifacts/{artifact_type}/{artifact_id}/publish",
    response_model=SpawnedJob,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
)
async def publish_artifact(
    request: PublishRequest,
    artifact_type: ArtifactType,
    artifact_id: str = Path(pattern=r"^[0-9a-f]{32}$"),
) -> SpawnedJob:
    try:
        job_id = await modal_client.spawn_publish(
            request,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
        )
    except Exception as exc:
        raise _spawn_error() from exc
    return SpawnedJob(job_id=job_id)
