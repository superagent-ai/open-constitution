from __future__ import annotations

import pytest

from activation_probe_mvp.data import Example, load_jsonl


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
