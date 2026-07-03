from __future__ import annotations

from types import SimpleNamespace

import torch

from activation_probe_mvp.exchange_classifier import decision_from_score
from activation_probe_mvp.modeling import LinearProbe
from scripts.guarded_generate import guarded_generate


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __call__(self, text, return_tensors):
        assert text == "User:\nprobe me\n\nAssistant:\n"
        assert return_tensors == "pt"
        return FakeBatch(input_ids=torch.tensor([[1]]))

    def decode(self, _token, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "held-token"


class HighRiskModel:
    def __call__(self, input_ids, output_hidden_states, use_cache):
        assert output_hidden_states is True
        assert use_cache is False
        return SimpleNamespace(
            logits=torch.tensor([[[0.0, 1.0]]]),
            hidden_states=[torch.full((1, input_ids.shape[1], 2), 10.0)],
        )


class BlockingClassifier:
    def classify(self, prompt, partial_response):
        assert prompt == "probe me"
        assert partial_response == "held-token"
        return decision_from_score(0.99, allow_threshold=0.3, block_threshold=0.7)


def test_guarded_generate_blocks_without_streaming_held_token(capsys):
    probe = LinearProbe(hidden_size=2)
    with torch.no_grad():
        probe.head.weight.fill_(1.0)
        probe.head.bias.zero_()

    result = guarded_generate(
        model=HighRiskModel(),
        tokenizer_or_processor=FakeTokenizer(),
        device="cpu",
        probe=probe,
        layer=0,
        prompt="probe me",
        max_new_tokens=1,
        probe_every_n_tokens=1,
        threshold=0.2,
        pause_threshold=0.5,
        smoothing_window=1,
        use_chat_template=False,
        classifier=BlockingClassifier(),
        refusal_message="refusal",
    )

    captured = capsys.readouterr()

    assert result["status"] == "blocked"
    assert result["response"] == "refusal"
    assert result["held_response_chars"] == len("held-token")
    assert "held-token" not in captured.out
    assert "refusal" in captured.out
