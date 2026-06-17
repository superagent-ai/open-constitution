from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    prompt: str
    response: str
    label: int


def load_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "response" not in obj or "label" not in obj:
                raise ValueError(f"Line {line_no} must contain prompt, response, label")
            label = int(obj["label"])
            if label not in (0, 1):
                raise ValueError(f"Line {line_no} label must be 0 or 1")
            examples.append(Example(prompt=obj["prompt"], response=obj["response"], label=label))
    return examples


def balanced_sample(
    examples: list[Example],
    *,
    max_examples: int | None,
    seed: int = 0,
) -> list[Example]:
    if max_examples is None or max_examples <= 0 or len(examples) <= max_examples:
        return examples

    by_label: dict[int, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    rng = random.Random(seed)
    for label_examples in by_label.values():
        rng.shuffle(label_examples)

    labels = sorted(by_label)
    base = max_examples // len(labels)
    remainder = max_examples % len(labels)

    sampled: list[Example] = []
    for i, label in enumerate(labels):
        take = base + (1 if i < remainder else 0)
        sampled.extend(by_label[label][:take])

    rng.shuffle(sampled)
    return sampled


def format_exchange(prompt: str, response: str) -> str:
    # Keep this simple for the MVP.
    # Later: use tokenizer.apply_chat_template for model-specific formatting.
    return f"User:\n{prompt}\n\nAssistant:\n{response}"
