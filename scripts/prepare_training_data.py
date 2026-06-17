#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


DEFAULT_PAIRWISE_DATASET_ID = "PKU-Alignment/PKU-SafeRLHF"
DEFAULT_SINGLE_RESPONSE_DATASET_ID = "PKU-Alignment/BeaverTails"
DEFAULT_DATASET_VERSION = "public_safety_v1"
DEFAULT_SINGLE_RESPONSE_MAX_RECORDS = 100_000
SCHEMA_FIELDS = [
    "id",
    "prompt",
    "response",
    "label",
    "split",
    "topic",
    "prompt_intent",
    "response_behavior",
    "dataset_version",
]


def normalize_split(split: str, *, prompt: str, validation_fraction: float) -> str:
    split = split.lower()
    if split in {"test", "eval", "evaluation"} or split.endswith("_test"):
        return "test"
    if split in {"dev", "val", "valid", "validation"} or split.endswith("_validation"):
        return "validation"
    if validation_fraction <= 0:
        return "train"

    # Split by prompt so response_0 and response_1 never leak across splits.
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < validation_fraction else "train"


def normalize_topic(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def parse_harm_categories(value: Any) -> dict[str, bool]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        return {}
    return {str(key): bool(val) for key, val in value.items()}


def topic_from_categories(value: Any) -> str:
    categories = parse_harm_categories(value)
    active = [name for name, enabled in categories.items() if enabled]
    if not active:
        return "safe"
    return normalize_topic(active[0])


def stable_key(record: dict[str, Any]) -> str:
    return hashlib.sha256(f"{record['prompt']}\0{record['response']}".encode("utf-8")).hexdigest()


def source_from_id(record_id: str) -> str:
    parts = record_id.split(":")
    return parts[1] if len(parts) >= 4 else "unknown"


def convert_pairwise_response(
    row: dict[str, Any],
    *,
    response_idx: int,
    source_split: str,
    record_index: int,
    dataset_version: str,
    source_name: str,
    validation_fraction: float,
) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    response = row.get(f"response_{response_idx}")
    is_safe = row.get(f"is_response_{response_idx}_safe")
    if not prompt or not response or not isinstance(is_safe, bool):
        return None

    label = 0 if is_safe else 1
    split = normalize_split(
        source_split,
        prompt=str(prompt),
        validation_fraction=validation_fraction,
    )
    topic = topic_from_categories(row.get(f"response_{response_idx}_harm_category"))
    if label == 1 and topic == "safe":
        topic = "policy_violation"

    return {
        "id": f"{dataset_version}:{source_name}:{split}:{record_index:06d}",
        "prompt": str(prompt),
        "response": str(response),
        "label": label,
        "split": split,
        "topic": topic,
        "prompt_intent": "unknown",
        "response_behavior": "allowed_response" if label == 0 else "policy_violation",
        "dataset_version": dataset_version,
    }


def convert_single_response_row(
    row: dict[str, Any],
    *,
    source_split: str,
    record_index: int,
    dataset_version: str,
    validation_fraction: float,
) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    response = row.get("response")
    is_safe = row.get("is_safe")
    if not prompt or not response or not isinstance(is_safe, bool):
        return None

    label = 0 if is_safe else 1
    split = normalize_split(
        source_split,
        prompt=str(prompt),
        validation_fraction=validation_fraction,
    )
    topic = topic_from_categories(row.get("category"))
    if label == 1 and topic == "safe":
        topic = "policy_violation"

    return {
        "id": f"{dataset_version}:single_response_safety:{split}:{record_index:06d}",
        "prompt": str(prompt),
        "response": str(response),
        "label": label,
        "split": split,
        "topic": topic,
        "prompt_intent": "unknown",
        "response_behavior": "allowed_response" if label == 0 else "policy_violation",
        "dataset_version": dataset_version,
    }


def iter_pairwise_records(
    dataset: DatasetDict,
    *,
    dataset_version: str,
    dedupe: bool,
    splits: Iterable[str] | None,
    validation_fraction: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    requested_splits = None if splits is None else set(splits)

    for source_split, split_dataset in dataset.items():
        if requested_splits is not None and source_split not in requested_splits:
            continue
        for row in split_dataset:
            for response_idx in (0, 1):
                record = convert_pairwise_response(
                    dict(row),
                    response_idx=response_idx,
                    source_split=source_split,
                    record_index=len(records),
                    dataset_version=dataset_version,
                    source_name="pairwise_safety",
                    validation_fraction=validation_fraction,
                )
                if record is None:
                    continue
                pair = (record["prompt"], record["response"])
                if dedupe and pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                records.append(record)

    return records


def iter_single_response_records(
    dataset: DatasetDict,
    *,
    dataset_version: str,
    dedupe: bool,
    splits: Iterable[str] | None,
    validation_fraction: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    requested_splits = None if splits is None else set(splits)

    for source_split, split_dataset in dataset.items():
        if requested_splits is not None and source_split not in requested_splits:
            continue
        for row in split_dataset:
            record = convert_single_response_row(
                dict(row),
                source_split=source_split,
                record_index=len(records),
                dataset_version=dataset_version,
                validation_fraction=validation_fraction,
            )
            if record is None:
                continue
            pair = (record["prompt"], record["response"])
            if dedupe and pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            records.append(record)

    return records


def cap_records_diverse(
    records: list[dict[str, Any]], max_records: int | None
) -> list[dict[str, Any]]:
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["label"], record["topic"])].append(record)

    for group_records in grouped.values():
        group_records.sort(key=stable_key)

    selected: list[dict[str, Any]] = []
    group_keys = sorted(grouped)
    while len(selected) < max_records and group_keys:
        next_keys: list[tuple[str, int, str]] = []
        for key in group_keys:
            if grouped[key]:
                selected.append(grouped[key].pop(0))
                if len(selected) == max_records:
                    break
            if grouped[key]:
                next_keys.append(key)
        group_keys = next_keys

    return selected


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for record in records:
        pair = (record["prompt"], record["response"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append(record)
    return deduped


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_summary(records: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    return {
        "file": str(output_path),
        "total_rows": len(records),
        "source_distribution": dict(
            sorted(Counter(source_from_id(r["id"]) for r in records).items())
        ),
        "label_distribution": dict(sorted(Counter(str(r["label"]) for r in records).items())),
        "split_distribution": dict(sorted(Counter(r["split"] for r in records).items())),
        "behavior_distribution": dict(
            sorted(Counter(r["response_behavior"] for r in records).items())
        ),
        "prompt_intent_distribution": dict(
            sorted(Counter(r["prompt_intent"] for r in records).items())
        ),
        "topic_distribution": dict(sorted(Counter(r["topic"] for r in records).items())),
        "topic_count": len({r["topic"] for r in records}),
        "unique_ids": len({r["id"] for r in records}),
        "unique_prompts": len({r["prompt"] for r in records}),
        "unique_prompt_response_pairs": len({(r["prompt"], r["response"]) for r in records}),
        "schema_fields": SCHEMA_FIELDS,
        "dataset_version": records[0]["dataset_version"] if records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert public safety datasets into Open Constitution JSONL."
    )
    parser.add_argument("--pairwise_dataset_id", default=DEFAULT_PAIRWISE_DATASET_ID)
    parser.add_argument("--single_response_dataset_id", default=DEFAULT_SINGLE_RESPONSE_DATASET_ID)
    parser.add_argument("--output_path", default="data/training_data.jsonl")
    parser.add_argument("--summary_path", default="data/dataset_summary.json")
    parser.add_argument("--dataset_version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument(
        "--single_response_max_records",
        type=int,
        default=DEFAULT_SINGLE_RESPONSE_MAX_RECORDS,
        help=(
            "Maximum single-response rows to include after deterministic label/topic-diverse "
            "sampling. Use 0 to include all rows."
        ),
    )
    parser.add_argument(
        "--no_single_response",
        action="store_true",
        help="Only convert the pairwise dataset.",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="Fraction of source train prompts assigned to validation.",
    )
    parser.add_argument(
        "--pairwise_splits",
        nargs="*",
        default=["train", "test"],
        help="Pairwise source splits to convert. Use all to convert every split.",
    )
    parser.add_argument(
        "--single_response_splits",
        nargs="*",
        default=["330k_train", "330k_test"],
        help="Single-response source splits to convert. Use all to convert every split.",
    )
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Keep duplicate prompt/response pairs instead of dropping them.",
    )
    args = parser.parse_args()

    if not 0 <= args.validation_fraction < 1:
        raise ValueError("--validation_fraction must be in [0, 1)")

    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)

    pairwise_dataset = load_dataset(args.pairwise_dataset_id)
    if not isinstance(pairwise_dataset, DatasetDict):
        pairwise_dataset = DatasetDict({"train": pairwise_dataset})
    records = iter_pairwise_records(
        pairwise_dataset,
        dataset_version=args.dataset_version,
        dedupe=not args.keep_duplicates,
        splits=None if args.pairwise_splits == ["all"] else args.pairwise_splits,
        validation_fraction=args.validation_fraction,
    )

    if not args.no_single_response:
        single_response_dataset = load_dataset(args.single_response_dataset_id)
        if not isinstance(single_response_dataset, DatasetDict):
            single_response_dataset = DatasetDict({"train": single_response_dataset})
        single_response_records = iter_single_response_records(
            single_response_dataset,
            dataset_version=args.dataset_version,
            dedupe=not args.keep_duplicates,
            splits=None if args.single_response_splits == ["all"] else args.single_response_splits,
            validation_fraction=args.validation_fraction,
        )
        records.extend(
            cap_records_diverse(
                single_response_records,
                None if args.single_response_max_records == 0 else args.single_response_max_records,
            )
        )

    if not args.keep_duplicates:
        records = dedupe_records(records)

    if not records:
        raise ValueError("No usable prompt/response rows were found.")

    write_jsonl(output_path, records)
    summary = build_summary(records, output_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
