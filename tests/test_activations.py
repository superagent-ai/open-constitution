from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from activation_probe_mvp import activations
from activation_probe_mvp.activations import get_selected_hidden_state


def test_get_selected_hidden_state_uses_hidden_states_first():
    hidden_states = [torch.tensor([0]), torch.tensor([1])]
    decoder_hidden_states = [torch.tensor([2])]
    outputs = SimpleNamespace(
        hidden_states=hidden_states,
        decoder_hidden_states=decoder_hidden_states,
    )

    assert get_selected_hidden_state(outputs, -1) is hidden_states[-1]


def test_get_selected_hidden_state_falls_back_to_decoder_hidden_states():
    decoder_hidden_states = [torch.tensor([0]), torch.tensor([1])]
    outputs = SimpleNamespace(hidden_states=None, decoder_hidden_states=decoder_hidden_states)

    assert get_selected_hidden_state(outputs, 0) is decoder_hidden_states[0]


def test_get_selected_hidden_state_errors_when_hidden_states_are_missing():
    outputs = SimpleNamespace(hidden_states=None, decoder_hidden_states=None)

    with pytest.raises(RuntimeError, match="did not include hidden_states"):
        get_selected_hidden_state(outputs, 0)


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"


class FakeModel:
    def __init__(self):
        self.moved_to = None
        self.eval_called = False

    def to(self, device):
        self.moved_to = device

    def eval(self):
        self.eval_called = True


def test_qwen_loader_uses_causal_lm_without_remote_code(monkeypatch):
    tokenizer_calls = []
    model_calls = []
    model = FakeModel()

    monkeypatch.setattr(
        activations.AutoConfig,
        "from_pretrained",
        lambda model_id, **kwargs: SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
    )
    monkeypatch.setattr(
        activations.AutoTokenizer,
        "from_pretrained",
        lambda model_id, **kwargs: tokenizer_calls.append((model_id, kwargs)) or FakeTokenizer(),
    )
    monkeypatch.setattr(
        activations.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, **kwargs: model_calls.append((model_id, kwargs)) or model,
    )

    loaded_model, tokenizer, device = activations.load_model_and_tokenizer(
        "Qwen/Qwen3-8B",
        device="cpu",
    )

    assert loaded_model is model
    assert tokenizer.pad_token == "<eos>"
    assert device == "cpu"
    assert tokenizer_calls[0][1]["trust_remote_code"] is False
    assert model_calls[0][1]["trust_remote_code"] is False
    assert model_calls[0][1]["device_map"] is None
    assert model.moved_to == "cpu"
    assert model.eval_called is True
