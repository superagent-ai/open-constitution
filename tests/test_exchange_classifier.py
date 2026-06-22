from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from activation_probe_mvp.exchange_classifier import (
    ExchangeClassifier,
    choose_verdict,
    decision_from_score,
)


class FakeTokenizer:
    def __init__(self):
        self.seen_text = None

    def __call__(self, text, return_tensors, truncation, max_length):
        self.seen_text = text
        assert return_tensors == "pt"
        assert truncation is True
        assert max_length == 16
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class FakeModel:
    def __init__(self, logits):
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, **_inputs):
        return SimpleNamespace(logits=self.logits)


def test_choose_verdict_maps_thresholds():
    assert choose_verdict(0.1, allow_threshold=0.3, block_threshold=0.7) == "allow"
    assert choose_verdict(0.5, allow_threshold=0.3, block_threshold=0.7) == "uncertain"
    assert choose_verdict(0.9, allow_threshold=0.3, block_threshold=0.7) == "block"


def test_choose_verdict_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="thresholds"):
        choose_verdict(0.5, allow_threshold=0.7, block_threshold=0.3)


def test_decision_from_score_sets_block_property():
    decision = decision_from_score(0.9, allow_threshold=0.3, block_threshold=0.7)

    assert decision.verdict == "block"
    assert decision.block is True


def test_exchange_classifier_scores_prompt_and_partial_response():
    tokenizer = FakeTokenizer()
    classifier = ExchangeClassifier(
        model=FakeModel([0.0, 2.0]),
        tokenizer=tokenizer,
        device="cpu",
        max_length=16,
        allow_threshold=0.3,
        block_threshold=0.7,
    )

    decision = classifier.classify("prompt", "partial")

    assert decision.verdict == "block"
    assert decision.score > 0.8
    assert tokenizer.seen_text == "User:\nprompt\n\nAssistant:\npartial"
