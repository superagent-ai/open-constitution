#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from activation_probe_mvp.activations import get_device
from activation_probe_mvp.data import load_jsonl
from activation_probe_mvp.exchange_classifier import ExchangeClassifier
from activation_probe_mvp.exchange_training import (
    DEFAULT_BLOCK_THRESHOLD,
    ClassifierRecord,
    build_classifier_records,
    records_to_dataset,
    split_records,
    tokenize_dataset,
)


def score_records(
    *,
    classifier: ExchangeClassifier,
    records: list[ClassifierRecord],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from transformers import DataCollatorWithPadding

    dataset = tokenize_dataset(
        records_to_dataset(records),
        classifier.tokenizer,
        max_length=classifier.max_length,
    )
    dataset = dataset.remove_columns(["text", "is_prefix"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=classifier.tokenizer),
    )
    scores: list[float] = []
    labels: list[int] = []

    classifier.model.eval()
    with torch.no_grad():
        for batch in loader:
            label_tensor = batch.pop("labels")
            inputs = {
                key: value.to(classifier.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            outputs = classifier.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores.extend(probs[:, classifier.unsafe_label_id].detach().cpu().tolist())
            labels.extend(label_tensor.tolist())

    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    positives = labels == 2 if 2 in labels else labels == 1
    preds = scores >= threshold
    safe_compliance = labels == 0
    safe_refusal = labels == 1 if 2 in labels else np.zeros_like(labels, dtype=bool)

    return {
        "block_threshold": round(float(threshold), 6),
        "precision": float(precision_score(positives, preds, zero_division=0)),
        "recall": float(recall_score(positives, preds, zero_division=0)),
        "f1": float(f1_score(positives, preds, zero_division=0)),
        "safe_compliance_block_rate": float(np.mean(preds[safe_compliance]))
        if np.any(safe_compliance)
        else None,
        "safe_refusal_block_rate": float(np.mean(preds[safe_refusal]))
        if np.any(safe_refusal)
        else None,
        "unsafe_block_rate": float(np.mean(preds[positives])) if np.any(positives) else None,
    }


def choose_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_safe_compliance_block_rate: float,
) -> dict[str, Any]:
    candidate_thresholds = sorted(
        set(np.linspace(0.01, 0.9999, 200).tolist() + scores.tolist()),
        reverse=True,
    )
    evaluated = [evaluate_threshold(scores, labels, threshold) for threshold in candidate_thresholds]
    feasible = [
        item
        for item in evaluated
        if item["safe_compliance_block_rate"] is None
        or item["safe_compliance_block_rate"] <= max_safe_compliance_block_rate
    ]
    pool = feasible or evaluated
    return max(
        pool,
        key=lambda item: (
            item["f1"],
            item["recall"],
            -float(item["safe_compliance_block_rate"] or 0.0),
            item["precision"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--max_safe_compliance_block_rate", type=float, default=0.02)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    examples = load_jsonl(args.data_path)
    records = build_classifier_records(examples)
    _, val_records = split_records(records, validation_size=args.validation_size, seed=args.seed)
    if args.max_records is not None:
        val_records = val_records[: args.max_records]

    classifier = ExchangeClassifier.from_pretrained(
        args.classifier_dir,
        device=args.device or get_device(),
        block_threshold=DEFAULT_BLOCK_THRESHOLD,
        max_length=args.max_length,
    )
    scores, labels = score_records(
        classifier=classifier,
        records=val_records,
        batch_size=args.batch_size,
    )
    best = choose_threshold(
        scores,
        labels,
        max_safe_compliance_block_rate=args.max_safe_compliance_block_rate,
    )
    result = {
        "classifier_dir": args.classifier_dir,
        "data_path": args.data_path,
        "validation_records": len(val_records),
        "label_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True), strict=True)
        },
        "max_safe_compliance_block_rate": args.max_safe_compliance_block_rate,
        "recommended": best,
        "default_0_65": evaluate_threshold(scores, labels, DEFAULT_BLOCK_THRESHOLD),
        "strict_0_9999": evaluate_threshold(scores, labels, 0.9999),
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.out_path is not None:
        Path(args.out_path).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
