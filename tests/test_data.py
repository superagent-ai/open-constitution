from __future__ import annotations

import pytest

from activation_probe_mvp.data import Example, balanced_sample, load_jsonl


def test_load_jsonl_reads_examples_and_skips_blank_lines(tmp_path):
    data_path = tmp_path / "examples.jsonl"
    data_path.write_text(
        '\n{"prompt": "hello", "response": "hi", "label": 0}\n'
        '{"prompt": "bad", "response": "blocked", "label": "1"}\n',
        encoding="utf-8",
    )

    assert load_jsonl(data_path) == [
        Example(prompt="hello", response="hi", label=0),
        Example(prompt="bad", response="blocked", label=1),
    ]


def test_load_jsonl_requires_expected_fields(tmp_path):
    data_path = tmp_path / "missing-field.jsonl"
    data_path.write_text('{"prompt": "hello", "label": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="prompt, response, label"):
        load_jsonl(data_path)


def test_load_jsonl_rejects_unknown_labels(tmp_path):
    data_path = tmp_path / "bad-label.jsonl"
    data_path.write_text(
        '{"prompt": "hello", "response": "hi", "label": 2}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label must be 0 or 1"):
        load_jsonl(data_path)


def test_balanced_sample_is_deterministic_and_balances_labels():
    examples = [
        Example(prompt=f"safe {i}", response="ok", label=0) for i in range(10)
    ] + [
        Example(prompt=f"unsafe {i}", response="bad", label=1) for i in range(10)
    ]

    sample_a = balanced_sample(examples, max_examples=6, seed=123)
    sample_b = balanced_sample(examples, max_examples=6, seed=123)

    assert sample_a == sample_b
    assert len(sample_a) == 6
    assert sum(example.label == 0 for example in sample_a) == 3
    assert sum(example.label == 1 for example in sample_a) == 3


def test_balanced_sample_returns_all_examples_when_not_capped():
    examples = [Example(prompt="hello", response="hi", label=0)]

    assert balanced_sample(examples, max_examples=None) == examples
    assert balanced_sample(examples, max_examples=0) == examples
    assert balanced_sample(examples, max_examples=10) == examples
