from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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


def format_exchange(prompt: str, response: str) -> str:
    # Keep this simple for the MVP.
    # Later: use tokenizer.apply_chat_template for model-specific formatting.
    return f"User:\n{prompt}\n\nAssistant:\n{response}"
