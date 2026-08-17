from __future__ import annotations

import pytest

from activation_probe_mvp.probe_models import (
    resolve_probe_model_route,
    validate_probe_model_id,
)


@pytest.mark.parametrize(
    ("model_id", "family", "gpu", "parameters_b"),
    [
        ("google/gemma-4-E2B-it", "gemma", "A10G", 2.0),
        ("Qwen/Qwen3-8B", "qwen", "L40S", 8.0),
        ("Qwen/Qwen3-32B", "qwen", "A100-80GB", 32.0),
        ("Qwen/Qwen3-235B-A22B", "qwen", "H200:4", 235.0),
        ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "kimi", "H200:2", 48.0),
        ("moonshotai/Kimi-K2-Instruct", "kimi", "B200:8", 1000.0),
        ("zai-org/GLM-4.7-Flash", "glm", "A100-80GB", 30.0),
        ("THUDM/glm-4-9b-chat", "glm", "L40S", 9.0),
        ("zai-org/GLM-5", "glm", "B200:8", 744.0),
    ],
)
def test_resolve_probe_model_route(model_id, family, gpu, parameters_b):
    route = resolve_probe_model_route(model_id)

    assert route.family == family
    assert route.gpu == gpu
    assert route.estimated_parameters_b == parameters_b


@pytest.mark.parametrize(
    "model_id",
    [
        "attacker/Qwen3-8B",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-8B-GGUF",
        "moonshotai/Kimi-K2.5",
        "zai-org/GLM-4V-9B",
        "google/paligemma-3b",
        "../escape",
    ],
)
def test_validate_probe_model_id_rejects_non_text_or_unofficial_models(model_id):
    with pytest.raises(ValueError):
        validate_probe_model_id(model_id)


def test_remote_code_is_limited_to_official_kimi_and_glm_models():
    assert resolve_probe_model_route("Qwen/Qwen3-8B").trust_remote_code is False
    assert resolve_probe_model_route("google/gemma-4-E2B-it").trust_remote_code is False
    assert resolve_probe_model_route("moonshotai/Kimi-K2-Instruct").trust_remote_code is True
    assert resolve_probe_model_route("zai-org/GLM-4.7-Flash").trust_remote_code is True
