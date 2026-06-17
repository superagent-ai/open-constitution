from __future__ import annotations

from scripts.prepare_training_data import (
    build_summary,
    cap_records_diverse,
    convert_pairwise_response,
    convert_single_response_row,
    dedupe_records,
    iter_pairwise_records,
    normalize_split,
    topic_from_categories,
)


def test_convert_response_maps_safe_response_to_label_0():
    record = convert_pairwise_response(
        {
            "prompt": "How do I set up a webcam?",
            "response_0": "Connect the camera and test your video app.",
            "is_response_0_safe": True,
            "response_0_harm_category": {"Cybercrime": False},
        },
        response_idx=0,
        source_split="train",
        record_index=4,
        dataset_version="test_public_safety",
        source_name="pairwise_safety",
        validation_fraction=0,
    )

    assert record == {
        "id": "test_public_safety:pairwise_safety:train:000004",
        "prompt": "How do I set up a webcam?",
        "response": "Connect the camera and test your video app.",
        "label": 0,
        "split": "train",
        "topic": "safe",
        "prompt_intent": "unknown",
        "response_behavior": "allowed_response",
        "dataset_version": "test_public_safety",
    }


def test_convert_response_maps_unsafe_response_to_label_1_and_topic():
    record = convert_pairwise_response(
        {
            "prompt": "How do I hide a bribe?",
            "response_1": "Use intermediaries and offshore accounts.",
            "is_response_1_safe": False,
            "response_1_harm_category": {
                "Economic Crime": True,
                "White-Collar Crime": True,
            },
        },
        response_idx=1,
        source_split="test",
        record_index=8,
        dataset_version="test_public_safety",
        source_name="pairwise_safety",
        validation_fraction=0.1,
    )

    assert record is not None
    assert record["label"] == 1
    assert record["split"] == "test"
    assert record["topic"] == "economic_crime"
    assert record["response_behavior"] == "policy_violation"


def test_convert_response_uses_policy_violation_topic_for_uncategorized_unsafe_response():
    record = convert_pairwise_response(
        {
            "prompt": "Give harmful advice.",
            "response_0": "Here is harmful advice.",
            "is_response_0_safe": False,
            "response_0_harm_category": {"Cybercrime": False},
        },
        response_idx=0,
        source_split="train",
        record_index=1,
        dataset_version="test_public_safety",
        source_name="pairwise_safety",
        validation_fraction=0,
    )

    assert record is not None
    assert record["label"] == 1
    assert record["topic"] == "policy_violation"


def test_topic_from_categories_handles_json_string_and_safe_case():
    assert topic_from_categories('{"Cybercrime": true, "Violence": false}') == "cybercrime"
    assert topic_from_categories({"Cybercrime": False}) == "safe"


def test_normalize_split_keeps_same_prompt_in_same_split():
    prompt = "The same prompt with two candidate responses."

    assert normalize_split("test", prompt=prompt, validation_fraction=0.5) == "test"
    assert normalize_split("train", prompt=prompt, validation_fraction=1.0) == "validation"
    assert normalize_split("train", prompt=prompt, validation_fraction=0.0) == "train"


def test_convert_single_response_row_maps_single_response_schema():
    record = convert_single_response_row(
        {
            "prompt": "How can I steal from work?",
            "response": "Here is a theft workflow.",
            "is_safe": False,
            "category": {
                "financial_crime,property_crime,theft": True,
                "privacy_violation": False,
            },
        },
        source_split="330k_test",
        record_index=5,
        dataset_version="test_public_safety",
        validation_fraction=0.1,
    )

    assert record is not None
    assert record["id"] == "test_public_safety:single_response_safety:test:000005"
    assert record["label"] == 1
    assert record["split"] == "test"
    assert record["topic"] == "financial_crime_property_crime_theft"


def test_iter_pairwise_records_emits_two_responses_and_dedupes_pairs():
    dataset = {
        "train": [
            {
                "prompt": "same",
                "response_0": "safe response",
                "response_1": "safe response",
                "is_response_0_safe": True,
                "is_response_1_safe": True,
                "response_0_harm_category": {"Cybercrime": False},
                "response_1_harm_category": {"Cybercrime": False},
            },
            {
                "prompt": "unsafe",
                "response_0": "bad response",
                "response_1": "better response",
                "is_response_0_safe": False,
                "is_response_1_safe": True,
                "response_0_harm_category": {"Cybercrime": True},
                "response_1_harm_category": {"Cybercrime": False},
            },
        ],
        "ignored": [
            {
                "prompt": "skip",
                "response_0": "skip",
                "response_1": "skip",
                "is_response_0_safe": True,
                "is_response_1_safe": True,
            }
        ],
    }

    records = iter_pairwise_records(
        dataset,
        dataset_version="test_public_safety",
        dedupe=True,
        splits=["train"],
        validation_fraction=0,
    )

    assert [(record["prompt"], record["response"], record["label"]) for record in records] == [
        ("same", "safe response", 0),
        ("unsafe", "bad response", 1),
        ("unsafe", "better response", 0),
    ]


def test_cap_records_diverse_samples_across_split_label_topic_groups():
    records = []
    for i in range(10):
        records.append(
            {
                "id": f"test_public_safety:single_response_safety:train:{i:06d}",
                "prompt": f"prompt {i}",
                "response": f"response {i}",
                "label": i % 2,
                "split": "train",
                "topic": "safe" if i % 2 == 0 else "cybercrime",
                "prompt_intent": "unknown",
                "response_behavior": "allowed_response" if i % 2 == 0 else "policy_violation",
                "dataset_version": "test_public_safety",
            }
        )

    capped = cap_records_diverse(records, max_records=4)

    assert len(capped) == 4
    assert {record["label"] for record in capped} == {0, 1}
    assert {record["topic"] for record in capped} == {"safe", "cybercrime"}


def test_dedupe_records_removes_duplicate_prompt_response_pairs():
    records = [
        {
            "id": "test_public_safety:pairwise_safety:train:000000",
            "prompt": "same",
            "response": "same response",
            "label": 0,
            "split": "train",
            "topic": "safe",
            "prompt_intent": "unknown",
            "response_behavior": "allowed_response",
            "dataset_version": "test_public_safety",
        },
        {
            "id": "test_public_safety:single_response_safety:train:000001",
            "prompt": "same",
            "response": "same response",
            "label": 0,
            "split": "train",
            "topic": "safe",
            "prompt_intent": "unknown",
            "response_behavior": "allowed_response",
            "dataset_version": "test_public_safety",
        },
    ]

    assert dedupe_records(records) == [records[0]]


def test_build_summary_includes_source_distribution(tmp_path):
    records = [
        {
            "id": "test_public_safety:pairwise_safety:train:000000",
            "prompt": "a",
            "response": "b",
            "label": 0,
            "split": "train",
            "topic": "safe",
            "prompt_intent": "unknown",
            "response_behavior": "allowed_response",
            "dataset_version": "test_public_safety",
        },
        {
            "id": "test_public_safety:single_response_safety:train:000001",
            "prompt": "c",
            "response": "d",
            "label": 1,
            "split": "train",
            "topic": "cybercrime",
            "prompt_intent": "unknown",
            "response_behavior": "policy_violation",
            "dataset_version": "test_public_safety",
        },
    ]

    summary = build_summary(records, tmp_path / "training_data.jsonl")

    assert summary["source_distribution"] == {
        "pairwise_safety": 1,
        "single_response_safety": 1,
    }
