#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REFUSAL_MARKERS = [
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "i am unable to",
    "i won't",
    "i will not",
    "i'm sorry",
    "i apologize",
    "can't provide",
    "cannot provide",
    "can't comply",
    "cannot comply",
]

LABEL_NAMES = {
    0: "safe_compliance",
    1: "safe_refusal",
    2: "unsafe_compliance",
}
LABEL_IDS = {value: key for key, value in LABEL_NAMES.items()}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_key(row: dict[str, Any]) -> str:
    return _sha256(
        json.dumps(
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "label": int(row["label"]),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def is_refusal(row: dict[str, Any]) -> bool:
    if row.get("judge_refusal") is True:
        return True
    behavior = str(row.get("response_behavior", "")).lower()
    if "refusal" in behavior or "refuse" in behavior:
        return True
    text = " ".join(str(row.get("response", "")).lower().split())
    return any(marker in text for marker in REFUSAL_MARKERS)


def three_class_label(row: dict[str, Any]) -> int:
    label_name = row.get("label_name")
    if label_name in LABEL_IDS:
        return LABEL_IDS[str(label_name)]

    label = int(row["label"])
    if label == 2:
        return 2
    if label == 1:
        return 1 if is_refusal(row) else 2
    if label == 0:
        return 1 if is_refusal(row) else 0
    raise ValueError(f"Unsupported label: {label}")


def convert_row(row: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    if "prompt" not in row or "response" not in row or "label" not in row:
        raise ValueError(f"{source_path} row is missing prompt, response, or label")

    converted = dict(row)
    label = three_class_label(row)
    converted["label"] = label
    converted["label_name"] = LABEL_NAMES[label]
    converted["three_class_source_path"] = str(source_path)
    return converted


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc


def build_dataset(
    *,
    primary_input_path: Path,
    extra_input_paths: list[Path],
    out_path: Path,
    keep_extra_duplicates: bool,
    extra_repeat: int,
) -> dict[str, Any]:
    if extra_repeat < 1:
        raise ValueError("extra_repeat must be at least 1")

    counts: Counter[str] = Counter()
    written = 0
    skipped_extra_duplicates = 0
    seen: set[str] = set()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        for row in iter_jsonl(primary_input_path):
            converted = convert_row(row, source_path=primary_input_path)
            seen.add(_row_key(converted))
            counts[converted["label_name"]] += 1
            written += 1
            out_file.write(json.dumps(converted, ensure_ascii=False) + "\n")

        for extra_path in extra_input_paths:
            for row in iter_jsonl(extra_path):
                converted = convert_row(row, source_path=extra_path)
                key = _row_key(converted)
                if not keep_extra_duplicates and key in seen:
                    skipped_extra_duplicates += 1
                    continue
                seen.add(key)
                for repeat_index in range(extra_repeat):
                    repeated = dict(converted)
                    repeated["extra_repeat_index"] = repeat_index
                    repeated["extra_repeat_count"] = extra_repeat
                    counts[repeated["label_name"]] += 1
                    written += 1
                    out_file.write(json.dumps(repeated, ensure_ascii=False) + "\n")

    return {
        "out_path": str(out_path),
        "primary_input_path": str(primary_input_path),
        "extra_input_paths": [str(path) for path in extra_input_paths],
        "rows": written,
        "counts": dict(counts),
        "skipped_extra_duplicates": skipped_extra_duplicates,
        "extra_repeat": extra_repeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_input_path", required=True)
    parser.add_argument("--extra_input_path", action="append", default=[])
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--keep_extra_duplicates", action="store_true")
    parser.add_argument("--extra_repeat", type=int, default=1)
    args = parser.parse_args()

    summary = build_dataset(
        primary_input_path=Path(args.primary_input_path),
        extra_input_paths=[Path(path) for path in args.extra_input_path],
        out_path=Path(args.out_path),
        keep_extra_duplicates=args.keep_extra_duplicates,
        extra_repeat=args.extra_repeat,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
