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


def test_load_jsonl_accepts_three_class_labels(tmp_path):
    data_path = tmp_path / "three-class.jsonl"
    data_path.write_text(
        '{"prompt": "hello", "response": "hi", "label": 2}\n',
        encoding="utf-8",
    )

    assert load_jsonl(data_path) == [
        Example(prompt="hello", response="hi", label=2),
    ]


def test_load_jsonl_rejects_unknown_labels(tmp_path):
    data_path = tmp_path / "bad-label.jsonl"
    data_path.write_text(
        '{"prompt": "hello", "response": "hi", "label": 3}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label must be 0, 1, or 2"):
        load_jsonl(data_path)


def test_balanced_sample_is_deterministic_and_balanced():
    examples = [Example(prompt=f"safe-{index}", response="ok", label=0) for index in range(10)] + [
        Example(prompt=f"unsafe-{index}", response="bad", label=1) for index in range(10)
    ]

    first = balanced_sample(examples, max_examples=8, seed=7)
    second = balanced_sample(examples, max_examples=8, seed=7)

    assert first == second
    assert [example.label for example in first].count(0) == 4
    assert [example.label for example in first].count(1) == 4


def test_balanced_sample_uses_available_examples_from_imbalanced_labels():
    examples = [
        Example(prompt="only-safe", response="ok", label=0),
        *[Example(prompt=f"unsafe-{index}", response="bad", label=1) for index in range(10)],
    ]

    sampled = balanced_sample(examples, max_examples=6, seed=0)

    assert len(sampled) == 6
    assert [example.label for example in sampled].count(0) == 1
    assert [example.label for example in sampled].count(1) == 5


def test_balanced_sample_validates_limit():
    with pytest.raises(ValueError, match="max_examples must be positive"):
        balanced_sample([], max_examples=0)
