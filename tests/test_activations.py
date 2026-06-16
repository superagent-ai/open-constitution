from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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
