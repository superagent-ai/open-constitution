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

    job_id = await modal_client.spawn_probe(
        ProbeTrainRequest(max_examples=50),
        artifact_id="a" * 32,
    )

    assert job_id == "fc-test"
    assert function.spawn_kwargs["data_path"] == "data/training_data.jsonl"
    assert function.spawn_kwargs["out_dir"] == f"jobs/{'a' * 32}/probe"
    assert function.spawn_kwargs["artifact_id"] == "a" * 32


@pytest.mark.anyio
async def test_spawn_classifier_uses_fixed_dataset_and_scoped_output(monkeypatch):
    function = FakeFunction()
    monkeypatch.setattr(modal_client, "_function", lambda _name: function)

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


class FakeFunctionCall:
    get_impl = FakeGet()

    def __init__(self):
        self.get = self.get_impl

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

    result = await modal_client.poll_job("fc-test")

    assert result["status"] == expected_status
    assert "sensitive remote details" not in repr(result)
