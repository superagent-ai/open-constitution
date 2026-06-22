from __future__ import annotations

import pytest

from activation_probe_mvp.data import Example
from activation_probe_mvp.exchange_training import (
    LFS_POINTER_PREFIX,
    build_classifier_records,
    format_classifier_input,
    split_records,
    trainer_processing_kwargs,
    validate_jsonl_source,
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
