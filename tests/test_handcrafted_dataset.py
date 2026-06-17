from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from activation_probe_mvp.data import load_jsonl


DATASET_PATH = Path("data/training_data.jsonl")
SUMMARY_PATH = Path("data/dataset_summary.json")
REQUIRED_FIELDS = {
    "id",
    "prompt",
    "response",
    "label",
    "split",
    "topic",
    "prompt_intent",
    "response_behavior",
    "dataset_version",
}


def load_records() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def source_from_id(record_id: str) -> str:
    parts = record_id.split(":")
    return parts[1] if len(parts) >= 4 else "unknown"


def test_training_dataset_shape_and_metadata():
    records = load_records()

    assert len(records) > 0
    assert all(REQUIRED_FIELDS <= record.keys() for record in records)
    assert {record["label"] for record in records} == {0, 1}
    assert len({record["id"] for record in records}) == len(records)


def test_training_dataset_distribution_metadata_is_present():
    records = load_records()

    assert Counter(record["split"] for record in records)
    assert Counter(record["response_behavior"] for record in records)
    assert Counter(record["topic"] for record in records)
    assert Counter(record["dataset_version"] for record in records)


def test_handcrafted_dataset_has_unique_prompt_response_pairs():
    records = load_records()
    pairs = {(record["prompt"], record["response"]) for record in records}

    assert len(pairs) == len(records)


def test_handcrafted_dataset_is_compatible_with_current_loader():
    examples = load_jsonl(DATASET_PATH)

    assert len(examples) == len(load_records())
    assert {example.label for example in examples} == {0, 1}


def test_dataset_summary_matches_training_data():
    records = load_records()
    summary = load_summary()

    assert summary["file"] == str(DATASET_PATH)
    assert summary["total_rows"] == len(records)
    assert summary["source_distribution"] == dict(
        sorted(Counter(source_from_id(record["id"]) for record in records).items())
    )
    assert summary["label_distribution"] == dict(
        sorted(Counter(str(record["label"]) for record in records).items())
    )
    assert summary["split_distribution"] == dict(
        sorted(Counter(record["split"] for record in records).items())
    )
    assert summary["behavior_distribution"] == dict(
        sorted(Counter(record["response_behavior"] for record in records).items())
    )
    assert summary["topic_distribution"] == dict(
        sorted(Counter(record["topic"] for record in records).items())
    )
    assert summary["unique_ids"] == len({record["id"] for record in records})
    assert summary["unique_prompt_response_pairs"] == len(
        {(record["prompt"], record["response"]) for record in records}
    )
