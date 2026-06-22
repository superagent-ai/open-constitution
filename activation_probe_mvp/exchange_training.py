from __future__ import annotations

import json
import random
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .chat import format_exchange
from .data import Example, load_jsonl

DEFAULT_CLASSIFIER_MODEL_ID = "answerdotai/ModernBERT-base"
DEFAULT_TRAINING_DATA_PATH = "data/training_data.jsonl"
DEFAULT_ALLOW_THRESHOLD = 0.35
DEFAULT_BLOCK_THRESHOLD = 0.65
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


@dataclass(frozen=True)
class ClassifierRecord:
    text: str
    label: int
    prompt: str
    response: str
    is_prefix: bool = False


def validate_jsonl_source(path: str | Path) -> Path:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Fetch or merge the Git LFS training dataset first."
        )

    with data_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if first_line == LFS_POINTER_PREFIX:
        raise ValueError(
            f"{data_path} is still a Git LFS pointer. Run `git lfs pull` before training."
        )

    return data_path


def format_classifier_input(prompt: str, response: str) -> str:
    return format_exchange(
        tokenizer_or_processor=None,
        prompt=prompt,
        response=response,
        use_chat_template=False,
    )


def make_response_prefix(
    response: str,
    rng: random.Random,
    min_ratio: float = 0.2,
    max_ratio: float = 0.8,
) -> str:
    if not response:
        return response

    if not 0 < min_ratio <= max_ratio <= 1:
        raise ValueError("prefix ratios must satisfy 0 < min_ratio <= max_ratio <= 1")

    ratio = rng.uniform(min_ratio, max_ratio)
    prefix_len = max(1, min(len(response), int(round(len(response) * ratio))))
    return response[:prefix_len]


def build_classifier_records(
    examples: list[Example],
    *,
    prefix_augment: bool = False,
    prefix_copies: int = 1,
    seed: int = 0,
) -> list[ClassifierRecord]:
    rng = random.Random(seed)
    records: list[ClassifierRecord] = []

    for ex in examples:
        records.append(
            ClassifierRecord(
                text=format_classifier_input(ex.prompt, ex.response),
                label=ex.label,
                prompt=ex.prompt,
                response=ex.response,
                is_prefix=False,
            )
        )

        if prefix_augment and ex.response:
            for _ in range(prefix_copies):
                prefix = make_response_prefix(ex.response, rng)
                records.append(
                    ClassifierRecord(
                        text=format_classifier_input(ex.prompt, prefix),
                        label=ex.label,
                        prompt=ex.prompt,
                        response=prefix,
                        is_prefix=True,
                    )
                )

    return records


def split_records(
    records: list[ClassifierRecord],
    *,
    validation_size: float = 0.2,
    seed: int = 0,
) -> tuple[list[ClassifierRecord], list[ClassifierRecord]]:
    if len(records) < 2:
        raise ValueError("Need at least two records for a train/validation split.")

    labels = [record.label for record in records]
    label_counts = {label: labels.count(label) for label in set(labels)}
    stratify = labels if len(label_counts) > 1 and min(label_counts.values()) >= 2 else None

    train_records, val_records = train_test_split(
        records,
        test_size=validation_size,
        random_state=seed,
        stratify=stratify,
    )

    return list(train_records), list(val_records)


def records_to_dataset(records: list[ClassifierRecord]):
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "text": record.text,
                "label": record.label,
                "is_prefix": record.is_prefix,
            }
            for record in records
        ]
    )


def tokenize_dataset(dataset, tokenizer, *, max_length: int):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_batch, batched=True)


def trainer_processing_kwargs(trainer_cls, tokenizer) -> dict[str, Any]:
    parameters = inspect.signature(trainer_cls.__init__).parameters

    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_classifier_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    labels = np.asarray(labels).astype(int)
    probs = _softmax(np.asarray(logits))[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    if len(set(labels.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))

    return metrics


def save_classifier_config(
    out_dir: str | Path,
    *,
    model_id: str,
    data_path: str | Path,
    max_length: int,
    allow_threshold: float,
    block_threshold: float,
    seed: int,
    metrics: dict[str, Any],
) -> None:
    config = {
        "model_id": model_id,
        "data_path": str(data_path),
        "max_length": max_length,
        "allow_threshold": allow_threshold,
        "block_threshold": block_threshold,
        "label_mapping": {"0": "safe", "1": "unsafe"},
        "seed": seed,
        "metrics": metrics,
        "text_format": "user_assistant",
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "classifier_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def train_exchange_classifier(
    *,
    model_id: str = DEFAULT_CLASSIFIER_MODEL_ID,
    data_path: str | Path = DEFAULT_TRAINING_DATA_PATH,
    out_dir: str | Path,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 512,
    validation_size: float = 0.2,
    seed: int = 0,
    prefix_augment: bool = False,
    prefix_copies: int = 1,
    allow_threshold: float = DEFAULT_ALLOW_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
    logging_steps: int = 500,
    save_steps: int = 5000,
    disable_tqdm: bool = True,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    if not 0 <= allow_threshold < block_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= allow_threshold < block_threshold <= 1")

    resolved_data_path = validate_jsonl_source(data_path)
    examples = load_jsonl(resolved_data_path)
    records = build_classifier_records(
        examples,
        prefix_augment=prefix_augment,
        prefix_copies=prefix_copies,
        seed=seed,
    )
    train_records, val_records = split_records(
        records,
        validation_size=validation_size,
        seed=seed,
    )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        trust_remote_code=True,
    )

    train_dataset = tokenize_dataset(
        records_to_dataset(train_records), tokenizer, max_length=max_length
    )
    val_dataset = tokenize_dataset(
        records_to_dataset(val_records), tokenizer, max_length=max_length
    )

    training_args_kwargs = {
        "output_dir": str(out_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 2,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "disable_tqdm": disable_tqdm,
        "load_best_model_at_end": False,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": [],
    }

    try:
        training_args = TrainingArguments(
            eval_strategy="no",
            **training_args_kwargs,
        )
    except TypeError:
        training_args = TrainingArguments(
            evaluation_strategy="no",
            **training_args_kwargs,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_classifier_metrics,
        **trainer_processing_kwargs(Trainer, tokenizer),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metadata = {
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "prefix_augment": prefix_augment,
        "prefix_copies": prefix_copies,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "eval": metrics,
    }

    save_classifier_config(
        out_dir,
        model_id=model_id,
        data_path=resolved_data_path,
        max_length=max_length,
        allow_threshold=allow_threshold,
        block_threshold=block_threshold,
        seed=seed,
        metrics=metadata,
    )

    return metadata
