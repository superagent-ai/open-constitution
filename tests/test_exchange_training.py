from __future__ import annotations

import json

import pytest

from activation_probe_mvp.data import Example
from activation_probe_mvp.exchange_training import (
    LFS_POINTER_PREFIX,
    build_classifier_records,
    format_classifier_input,
    label_mapping_for_examples,
    split_records,
    trainer_processing_kwargs,
    validate_jsonl_source,
    write_training_progress,
)


def test_format_classifier_input_uses_simple_exchange_format():
    assert format_classifier_input("Explain", "Answer") == "User:\nExplain\n\nAssistant:\nAnswer"


def test_validate_jsonl_source_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Git LFS training dataset"):
        validate_jsonl_source(tmp_path / "missing.jsonl")


def test_validate_jsonl_source_rejects_lfs_pointer(tmp_path):
    data_path = tmp_path / "training_data.jsonl"
    data_path.write_text(
        f"{LFS_POINTER_PREFIX}\noid sha256:34c1a80368\nsize 211000000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Git LFS pointer"):
        validate_jsonl_source(data_path)


def test_write_training_progress_atomically_adds_timestamp(tmp_path):
    progress_path = tmp_path / "progress.json"

    write_training_progress(
        progress_path,
        {
            "phase": "training",
            "step": 100,
            "total_steps": 400,
            "percent": 25.0,
        },
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["phase"] == "training"
    assert progress["step"] == 100
    assert progress["percent"] == 25.0
    assert progress["updated_at"].endswith("+00:00")
    assert not (tmp_path / "progress.json.tmp").exists()


def test_build_classifier_records_adds_deterministic_prefix_examples():
    examples = [Example(prompt="p", response="abcdefghi", label=1)]

    records = build_classifier_records(
        examples,
        prefix_augment=True,
        prefix_copies=2,
        seed=7,
    )

    assert len(records) == 3
    assert records[0].is_prefix is False
    assert [record.is_prefix for record in records[1:]] == [True, True]
    assert all(record.label == 1 for record in records)
    assert all(0 < len(record.response) < len(examples[0].response) for record in records[1:])


def test_split_records_stratifies_when_each_label_has_enough_examples():
    examples = [Example(prompt=f"safe-{i}", response="ok", label=0) for i in range(4)] + [
        Example(prompt=f"unsafe-{i}", response="bad", label=1) for i in range(4)
    ]
    records = build_classifier_records(examples)

    train_records, val_records = split_records(records, validation_size=0.25, seed=0)

    assert len(train_records) == 6
    assert len(val_records) == 2
    assert {record.label for record in val_records} == {0, 1}


def test_label_mapping_for_examples_supports_three_class_labels():
    examples = [
        Example(prompt="safe", response="ok", label=0),
        Example(prompt="refusal", response="I can't help with that.", label=1),
        Example(prompt="unsafe", response="bad", label=2),
    ]

    assert label_mapping_for_examples(examples) == {
        "0": "safe_compliance",
        "1": "safe_refusal",
        "2": "unsafe_compliance",
    }


def test_trainer_processing_kwargs_supports_transformers_api_names():
    tokenizer = object()

    class NewTrainer:
        def __init__(self, processing_class):
            self.processing_class = processing_class

    class OldTrainer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    assert trainer_processing_kwargs(NewTrainer, tokenizer) == {"processing_class": tokenizer}
    assert trainer_processing_kwargs(OldTrainer, tokenizer) == {"tokenizer": tokenizer}
