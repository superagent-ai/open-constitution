from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ProbeFamily = Literal["gemma", "qwen", "kimi", "glm"]

DEFAULT_PROBE_MODEL_ID = "google/gemma-4-E2B-it"

_SEGMENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_MODEL_ID_PATTERN = re.compile(rf"^{_SEGMENT}/{_SEGMENT}$")
_SIZE_PATTERN = re.compile(r"(?:^|[-_])(?:[EA])?(\d+(?:\.\d+)?)B(?:$|[-_])", re.IGNORECASE)
_UNSUPPORTED_NAME_PATTERN = re.compile(
    r"(?:^|[-_.])(?:"
    r"audio|embedding|embed|image|omni|reranker|reward|vision|vl|"
    r"asr|ocr|gguf|awq|gptq|mlx"
    r")(?:$|[-_.])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProbeModelRoute:
    model_id: str
    family: ProbeFamily
    gpu: str
    memory_mib: int
    estimated_parameters_b: float
    trust_remote_code: bool

    @property
    def multi_gpu(self) -> bool:
        return ":" in self.gpu


def _family_for_model_id(model_id: str) -> ProbeFamily:
    namespace, name = model_id.split("/", 1)
    lowered_name = name.lower()

    if namespace == "google" and lowered_name.startswith("gemma-"):
        return "gemma"
    if namespace == "Qwen" and lowered_name.startswith("qwen"):
        return "qwen"
    if namespace == "moonshotai" and lowered_name.startswith("kimi"):
        return "kimi"
    if namespace in {"zai-org", "THUDM"} and "glm" in lowered_name:
        return "glm"
    raise ValueError("model_id must be an official Gemma, Qwen, Kimi, or GLM checkpoint")


def validate_probe_model_id(model_id: str) -> str:
    if _MODEL_ID_PATTERN.fullmatch(model_id) is None or ".." in model_id or "--" in model_id:
        raise ValueError("model_id must use the form 'official-namespace/model-name'")

    family = _family_for_model_id(model_id)
    name = model_id.split("/", 1)[1]
    lowered_name = name.lower()

    if _UNSUPPORTED_NAME_PATTERN.search(name):
        raise ValueError("model_id must be a text-generation checkpoint")
    if family == "gemma" and "paligemma" in lowered_name:
        raise ValueError("PaliGemma checkpoints are not supported for text probes")
    if family == "glm" and re.search(r"glm-?4v", lowered_name):
        raise ValueError("GLM vision checkpoints are not supported for text probes")
    if family == "kimi" and any(
        version in lowered_name for version in ("k2.5", "k2-5", "k2.6", "k2-6")
    ):
        raise ValueError("multimodal Kimi checkpoints are not supported for text probes")

    return model_id


def _estimate_parameters_b(model_id: str, family: ProbeFamily) -> float:
    name = model_id.split("/", 1)[1]
    sizes = [float(value) for value in _SIZE_PATTERN.findall(name)]
    if sizes:
        return max(sizes)

    lowered_name = name.lower()
    if family == "gemma":
        return 4.0
    if family == "qwen":
        return 80.0
    if family == "kimi":
        if "k2" in lowered_name:
            return 1000.0
        return 80.0
    if "glm-5" in lowered_name or "glm5" in lowered_name:
        return 744.0
    if "flash" in lowered_name:
        return 30.0
    if "air" in lowered_name:
        return 106.0
    return 355.0


def _gpu_for_parameters(parameters_b: float) -> str:
    if parameters_b <= 4:
        return "A10G"
    if parameters_b <= 12:
        return "L40S"
    if parameters_b <= 34:
        return "A100-80GB"
    if parameters_b <= 80:
        return "H200:2"
    if parameters_b <= 250:
        return "H200:4"
    return "B200:8"


def _memory_for_parameters(parameters_b: float) -> int:
    if parameters_b <= 12:
        return 32768
    if parameters_b <= 80:
        return 65536
    if parameters_b <= 250:
        return 131072
    return 262144


def resolve_probe_model_route(model_id: str) -> ProbeModelRoute:
    validated_id = validate_probe_model_id(model_id)
    family = _family_for_model_id(validated_id)
    parameters_b = _estimate_parameters_b(validated_id, family)
    return ProbeModelRoute(
        model_id=validated_id,
        family=family,
        gpu=_gpu_for_parameters(parameters_b),
        memory_mib=_memory_for_parameters(parameters_b),
        estimated_parameters_b=parameters_b,
        trust_remote_code=family in {"kimi", "glm"},
    )
