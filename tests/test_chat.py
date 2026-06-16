from __future__ import annotations

from activation_probe_mvp.chat import format_exchange, format_generation_prompt


class TemplateOwner:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        roles = ",".join(message["role"] for message in messages)
        return f"template:{roles}:{add_generation_prompt}"


class LegacyTemplateOwner:
    def apply_chat_template(self, messages, tokenize=False):
        roles = ",".join(message["role"] for message in messages)
        return f"legacy:{roles}:{tokenize}"


class ProcessorWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_format_exchange_falls_back_without_chat_template():
    assert (
        format_exchange(object(), prompt="Explain", response="Answer")
        == "User:\nExplain\n\nAssistant:\nAnswer"
    )


def test_format_generation_prompt_falls_back_without_chat_template():
    assert format_generation_prompt(object(), prompt="Explain") == "User:\nExplain\n\nAssistant:\n"


def test_format_exchange_uses_direct_chat_template_owner():
    owner = TemplateOwner()

    assert (
        format_exchange(owner, prompt="Explain", response="Answer")
        == "template:user,assistant:False"
    )
    assert owner.calls == [
        {
            "messages": [
                {"role": "user", "content": "Explain"},
                {"role": "assistant", "content": "Answer"},
            ],
            "tokenize": False,
            "add_generation_prompt": False,
        }
    ]


def test_format_generation_prompt_uses_nested_tokenizer_template():
    owner = TemplateOwner()
    processor = ProcessorWrapper(owner)

    assert format_generation_prompt(processor, prompt="Explain") == "template:user:True"
    assert owner.calls[0]["messages"] == [{"role": "user", "content": "Explain"}]
    assert owner.calls[0]["add_generation_prompt"] is True


def test_format_exchange_supports_legacy_template_signature():
    assert (
        format_exchange(LegacyTemplateOwner(), prompt="Explain", response="Answer")
        == "legacy:user,assistant:False"
    )
