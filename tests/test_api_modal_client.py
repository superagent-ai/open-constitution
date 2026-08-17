from __future__ import annotations

import pytest

from api import modal_client
from api.schemas import ClassifierTrainRequest, ProbeTrainRequest, PublishRequest


class FakeSpawn:
    def __init__(self, owner):
        self.owner = owner

    async def aio(self, **kwargs):
        self.owner.spawn_kwargs = kwargs
        return type("Call", (), {"object_id": "fc-test"})()


class FakeFunction:
    def __init__(self):
        self.spawn = FakeSpawn(self)
        self.spawn_kwargs = None
        self.options = None

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


@pytest.mark.anyio
async def test_spawn_probe_uses_fixed_dataset_and_scoped_output(monkeypatch):
    function = FakeFunction()
    monkeypatch.setattr(modal_client, "_function", lambda _name: function)

    async def ignore_registration(_job_id, _artifact_id, **_kwargs):
        return None

    monkeypatch.setattr(modal_client, "_register_job", ignore_registration)

    job_id = await modal_client.spawn_probe(
        ProbeTrainRequest(model_id="Qwen/Qwen3-8B", max_examples=50),
        artifact_id="a" * 32,
    )

    assert job_id == "fc-test"
    assert function.spawn_kwargs["data_path"] == "data/training_data.jsonl"
    assert function.spawn_kwargs["model_id"] == "Qwen/Qwen3-8B"
    assert function.spawn_kwargs["out_dir"] == f"jobs/{'a' * 32}/probe"
    assert function.spawn_kwargs["artifact_id"] == "a" * 32
    assert function.options == {"gpu": "L40S", "memory": 32768}


@pytest.mark.anyio
async def test_spawn_classifier_uses_fixed_dataset_and_scoped_output(monkeypatch):
    function = FakeFunction()
    monkeypatch.setattr(modal_client, "_function", lambda _name: function)

    async def ignore_registration(_job_id, _artifact_id, **_kwargs):
        return None

    monkeypatch.setattr(modal_client, "_register_job", ignore_registration)

    job_id = await modal_client.spawn_classifier(
        ClassifierTrainRequest(epochs=1),
        artifact_id="c" * 32,
    )

    assert job_id == "fc-test"
    assert function.spawn_kwargs["data_path"] == "data/training_data_classifier.jsonl"
    assert function.spawn_kwargs["out_dir"] == f"jobs/{'c' * 32}/classifier"
    assert function.spawn_kwargs["artifact_id"] == "c" * 32


@pytest.mark.anyio
async def test_publish_token_is_in_dynamic_secret_not_function_arguments(monkeypatch):
    function = FakeFunction()
    monkeypatch.setattr(modal_client, "_function", lambda _name: function)

    class FakeSecret:
        @staticmethod
        def from_dict(values):
            return {"secret_values": values}

    monkeypatch.setattr(modal_client.modal, "Secret", FakeSecret)
    request = PublishRequest(repo_id="owner/model", hf_token="hf_user_secret")

    job_id = await modal_client.spawn_publish(
        request,
        artifact_id="b" * 32,
        artifact_type="classifier",
    )

    assert job_id == "fc-test"
    assert function.options == {"secrets": [{"secret_values": {"HF_TOKEN": "hf_user_secret"}}]}
    assert "hf_user_secret" not in repr(function.spawn_kwargs)
    assert "hf_token" not in function.spawn_kwargs


class FakeGet:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def aio(self, *, timeout):
        assert timeout == 0
        if self.error is not None:
            raise self.error
        return self.result


class FakeCancel:
    def __init__(self):
        self.called = False

    async def aio(self):
        self.called = True


class FakeFunctionCall:
    get_impl = FakeGet()
    cancel_impl = FakeCancel()

    def __init__(self):
        self.get = self.get_impl
        self.cancel = self.cancel_impl

    @classmethod
    def from_id(cls, _job_id):
        return cls()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("get_impl", "expected_status"),
    [
        (FakeGet(error=TimeoutError()), "running"),
        (FakeGet(error=modal_client.OutputExpiredError("expired")), "expired"),
        (FakeGet(result={"artifact_id": "abc"}), "succeeded"),
        (FakeGet(error=RuntimeError("sensitive remote details")), "failed"),
    ],
)
async def test_poll_job_maps_modal_state(monkeypatch, get_impl, expected_status):
    FakeFunctionCall.get_impl = get_impl
    monkeypatch.setattr(modal_client.modal, "FunctionCall", FakeFunctionCall)

    async def no_progress(_job_id):
        return None

    monkeypatch.setattr(modal_client, "_job_progress", no_progress)

    result = await modal_client.poll_job("fc-test")

    assert result["status"] == expected_status
    assert "sensitive remote details" not in repr(result)


@pytest.mark.anyio
async def test_poll_running_job_includes_progress(monkeypatch):
    FakeFunctionCall.get_impl = FakeGet(error=TimeoutError())
    monkeypatch.setattr(modal_client.modal, "FunctionCall", FakeFunctionCall)

    async def current_progress(_job_id):
        return {
            "phase": "training",
            "step": 100,
            "total_steps": 400,
            "percent": 25.0,
            "eta_seconds": 900,
        }

    monkeypatch.setattr(modal_client, "_job_progress", current_progress)

    result = await modal_client.poll_job("fc-test")

    assert result["status"] == "running"
    assert result["progress"]["percent"] == 25.0
    assert result["progress"]["eta_seconds"] == 900


class FakeDictMethod:
    def __init__(self, values, operation):
        self.values = values
        self.operation = operation

    async def aio(self, key, default=None):
        if self.operation == "get":
            return self.values.get(key, default)
        return self.values.pop(key, default)


class FakeProgressStore:
    def __init__(self, values):
        self.values = values
        self.get = FakeDictMethod(values, "get")
        self.pop = FakeDictMethod(values, "pop")


class FakeRemoveFile:
    def __init__(self):
        self.calls = []
        self.error = None

    async def aio(self, path, *, recursive):
        self.calls.append((path, recursive))
        if self.error is not None:
            raise self.error


class FakeVolume:
    remove_file = FakeRemoveFile()

    @classmethod
    def from_name(cls, _name, *, environment_name=None):
        return cls()


@pytest.mark.anyio
async def test_cancel_job_terminates_container_and_removes_artifacts(monkeypatch):
    artifact_id = "d" * 32
    store = FakeProgressStore(
        {
            "job:fc-test": artifact_id,
            f"progress:{artifact_id}": {"percent": 25.0},
        }
    )
    FakeFunctionCall.cancel_impl = FakeCancel()
    FakeVolume.remove_file = FakeRemoveFile()
    monkeypatch.setattr(modal_client, "_progress_store", lambda: store)
    monkeypatch.setattr(modal_client.modal, "FunctionCall", FakeFunctionCall)
    monkeypatch.setattr(modal_client.modal, "Volume", FakeVolume)

    result = await modal_client.cancel_job("fc-test")

    assert FakeFunctionCall.cancel_impl.called is True
    assert FakeVolume.remove_file.calls == [(f"jobs/{artifact_id}", True)]
    assert store.values == {}
    assert result == {
        "job_id": "fc-test",
        "artifact_id": artifact_id,
        "status": "cancelled",
        "execution_cancelled": True,
        "artifacts_removed": True,
        "cleanup_errors": [],
    }


@pytest.mark.anyio
async def test_cancel_job_returns_none_without_registered_artifact(monkeypatch):
    monkeypatch.setattr(modal_client, "_progress_store", lambda: FakeProgressStore({}))

    assert await modal_client.cancel_job("fc-missing") is None


@pytest.mark.anyio
async def test_cancel_job_treats_missing_artifact_directory_as_clean(monkeypatch):
    artifact_id = "e" * 32
    store = FakeProgressStore(
        {
            "job:fc-test": artifact_id,
            f"progress:{artifact_id}": {"percent": 0.0},
        }
    )
    FakeFunctionCall.cancel_impl = FakeCancel()
    FakeVolume.remove_file = FakeRemoveFile()
    FakeVolume.remove_file.error = modal_client.InvalidError("No such file or directory.")
    monkeypatch.setattr(modal_client, "_progress_store", lambda: store)
    monkeypatch.setattr(modal_client.modal, "FunctionCall", FakeFunctionCall)
    monkeypatch.setattr(modal_client.modal, "Volume", FakeVolume)

    result = await modal_client.cancel_job("fc-test")

    assert result["artifacts_removed"] is True
    assert result["cleanup_errors"] == []
    assert store.values == {}
