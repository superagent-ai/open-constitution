from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .activations import get_device
from .exchange_training import (
    DEFAULT_ALLOW_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
    format_classifier_input,
)

Verdict = Literal["allow", "uncertain", "block"]
UNSAFE_LABEL_NAMES = {"unsafe", "unsafe_compliance"}

DEFAULT_REFUSAL = (
    "I can't help with that request. I can help with safe, lawful alternatives instead."
)


@dataclass(frozen=True)
class ExchangeDecision:
    verdict: Verdict
    score: float
    reason: str
    allow_threshold: float
    block_threshold: float
    label_scores: dict[str, float] | None = None

    @property
    def block(self) -> bool:
        return self.verdict == "block"


def choose_verdict(
    score: float,
    *,
    allow_threshold: float = DEFAULT_ALLOW_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
) -> Verdict:
    if not 0 <= allow_threshold < block_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= allow_threshold < block_threshold <= 1")

    if score >= block_threshold:
        return "block"
    if score >= allow_threshold:
        return "uncertain"
    return "allow"


def decision_from_score(
    score: float,
    *,
    allow_threshold: float = DEFAULT_ALLOW_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
    reason: str = "modernbert_exchange_classifier",
    label_scores: dict[str, float] | None = None,
) -> ExchangeDecision:
    return ExchangeDecision(
        verdict=choose_verdict(
            score,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
        ),
        score=float(score),
        reason=reason,
        allow_threshold=allow_threshold,
        block_threshold=block_threshold,
        label_scores=label_scores,
    )


def load_classifier_config(classifier_dir: str | Path) -> dict:
    config_path = Path(classifier_dir) / "classifier_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


class ExchangeClassifier:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: str,
        max_length: int = 512,
        allow_threshold: float = DEFAULT_ALLOW_THRESHOLD,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        label_mapping: dict[str, str] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold
        self.label_mapping = label_mapping or self._infer_label_mapping()
        self.unsafe_label_id = self._unsafe_label_id()

    @classmethod
    def from_pretrained(
        cls,
        classifier_dir: str | Path,
        *,
        device: str | None = None,
        allow_threshold: float | None = None,
        block_threshold: float | None = None,
        max_length: int | None = None,
    ) -> "ExchangeClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config = load_classifier_config(classifier_dir)
        device = device or get_device()
        resolved_max_length = int(max_length or config.get("max_length", 512))
        resolved_allow_threshold = float(
            allow_threshold
            if allow_threshold is not None
            else config.get("allow_threshold", DEFAULT_ALLOW_THRESHOLD)
        )
        resolved_block_threshold = float(
            block_threshold
            if block_threshold is not None
            else config.get("block_threshold", DEFAULT_BLOCK_THRESHOLD)
        )

        tokenizer = AutoTokenizer.from_pretrained(classifier_dir, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            classifier_dir,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=resolved_max_length,
            allow_threshold=resolved_allow_threshold,
            block_threshold=resolved_block_threshold,
            label_mapping=config.get("label_mapping"),
        )

    def _infer_label_mapping(self) -> dict[str, str]:
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if id2label:
            return {str(key): str(value) for key, value in id2label.items()}

        num_labels = int(getattr(getattr(self.model, "config", None), "num_labels", 2))
        if num_labels == 3:
            return {"0": "safe_compliance", "1": "safe_refusal", "2": "unsafe_compliance"}
        return {"0": "safe", "1": "unsafe"}

    def _unsafe_label_id(self) -> int:
        for label_id, label_name in self.label_mapping.items():
            if label_name in UNSAFE_LABEL_NAMES:
                return int(label_id)

        # Fall back to the last class so older ad-hoc checkpoints still score the risky class.
        return max(int(label_id) for label_id in self.label_mapping)

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        if hasattr(inputs, "to"):
            return inputs.to(self.device)

        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @torch.no_grad()
    def scores(self, prompt: str, partial_response: str) -> dict[str, float]:
        text = format_classifier_input(prompt, partial_response)
        inputs = self._tokenize(text)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return {
            self.label_mapping.get(str(label_id), str(label_id)): float(probs[:, label_id].item())
            for label_id in range(probs.shape[-1])
        }

    @torch.no_grad()
    def score(self, prompt: str, partial_response: str) -> float:
        scores = self.scores(prompt, partial_response)
        unsafe_label = self.label_mapping.get(str(self.unsafe_label_id), str(self.unsafe_label_id))
        return scores[unsafe_label]

    def classify(self, prompt: str, partial_response: str) -> ExchangeDecision:
        scores = self.scores(prompt, partial_response)
        unsafe_label = self.label_mapping.get(str(self.unsafe_label_id), str(self.unsafe_label_id))
        return decision_from_score(
            scores[unsafe_label],
            allow_threshold=self.allow_threshold,
            block_threshold=self.block_threshold,
            reason=f"modernbert_exchange_classifier:{unsafe_label}",
            label_scores=scores,
        )


def classify_exchange(
    prompt: str,
    partial_response: str,
    *,
    classifier_dir: str | Path,
    device: str | None = None,
    allow_threshold: float | None = None,
    block_threshold: float | None = None,
    max_length: int | None = None,
) -> ExchangeDecision:
    classifier = ExchangeClassifier.from_pretrained(
        classifier_dir,
        device=device,
        allow_threshold=allow_threshold,
        block_threshold=block_threshold,
        max_length=max_length,
    )
    return classifier.classify(prompt, partial_response)
