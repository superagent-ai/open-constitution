#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from scripts.benchmark_jailbreak import (
    OPENROUTER_DEFAULT_MODEL,
    OpenRouterJudge,
    _sha256_short,
    load_env_file,
    teacher_label_from_judge,
)
from scripts.build_benign_recovery_data import generate_safe_response, judge_safe_compliance


def prompt_ok(prompt: str, *, max_prompt_chars: int) -> bool:
    stripped = prompt.strip()
    if len(stripped) < 8:
        return False
    if len(stripped) > max_prompt_chars:
        return False
    return True


def write_record(out_file, record: dict[str, Any]) -> None:
    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def toxicchat_source_record(
    *,
    row: dict[str, Any],
    response: str,
    label: int,
    label_name: str,
    response_kind: str,
    judge: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    return {
        "prompt": str(row["user_input"]),
        "response": response,
        "label": label,
        "label_name": label_name,
        "source": "ToxicChat augmentation",
        "dataset": "lmsys/toxic-chat",
        "dataset_config": "toxicchat0124",
        "dataset_split": "train",
        "response_kind": response_kind,
        "conv_id": str(row.get("conv_id", "")),
        "toxicity": int(row.get("toxicity", 0)),
        "jailbreaking": int(row.get("jailbreaking", 0)),
        "generator_or_source_model": model,
        "judge_type": judge.get("judge_type", "openrouter"),
        "judge_model": judge.get("model"),
        "judge_unsafe": bool(judge.get("unsafe", False)),
        "judge_refusal": bool(judge.get("refusal", False)),
        "judge_confidence": float(judge.get("confidence", 0.0)),
        "judge_category": judge.get("category"),
        "judge_rationale": str(judge.get("rationale", ""))[:240],
        "prompt_hash": _sha256_short(str(row["user_input"])),
        "response_hash": _sha256_short(response),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--dataset_config", default="toxicchat0124")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_safe_compliance", type=int, default=300)
    parser.add_argument("--max_judged_model_outputs", type=int, default=150)
    parser.add_argument("--max_prompt_chars", type=int, default=2200)
    parser.add_argument("--generator_model", default=OPENROUTER_DEFAULT_MODEL)
    parser.add_argument("--judge_model", default=OPENROUTER_DEFAULT_MODEL)
    parser.add_argument("--openrouter_api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--timeout_seconds", type=float, default=60)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    load_env_file()
    api_key = os.environ.get(args.openrouter_api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.openrouter_api_key_env} is not set.")

    dataset = list(load_dataset("lmsys/toxic-chat", args.dataset_config, split=args.split))
    rng = random.Random(args.seed)
    rng.shuffle(dataset)

    safe_rows = [
        row
        for row in dataset
        if int(row.get("toxicity", 0)) == 0
        and int(row.get("jailbreaking", 0)) == 0
        and prompt_ok(str(row["user_input"]), max_prompt_chars=args.max_prompt_chars)
    ]
    risk_rows = [
        row
        for row in dataset
        if (int(row.get("toxicity", 0)) == 1 or int(row.get("jailbreaking", 0)) == 1)
        and prompt_ok(str(row["user_input"]), max_prompt_chars=args.max_prompt_chars)
        and str(row.get("model_output", "")).strip()
    ]

    judge = OpenRouterJudge(
        model=args.judge_model,
        api_key_env=args.openrouter_api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"safe_compliance": 0, "safe_refusal": 0, "unsafe_compliance": 0}
    skipped = {"safe_generation_or_judge": 0, "risk_judge": 0}

    with out_path.open("w", encoding="utf-8") as out_file:
        for row in safe_rows:
            if counts["safe_compliance"] >= args.max_safe_compliance:
                break
            prompt = str(row["user_input"])
            response = generate_safe_response(
                prompt=prompt,
                model=args.generator_model,
                api_key=api_key,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            compliance_judge = judge_safe_compliance(
                prompt=prompt,
                response=response,
                model=args.judge_model,
                api_key=api_key,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            if (
                not bool(compliance_judge.get("safe_compliance"))
                or bool(compliance_judge.get("unsafe"))
                or bool(compliance_judge.get("refusal"))
            ):
                skipped["safe_generation_or_judge"] += 1
                continue
            record = toxicchat_source_record(
                row=row,
                response=response,
                label=0,
                label_name="safe_compliance",
                response_kind="generated_safe_compliance",
                judge={
                    "judge_type": "openrouter_safe_compliance",
                    "model": args.judge_model,
                    "unsafe": False,
                    "refusal": False,
                    "confidence": float(compliance_judge.get("confidence", 0.0)),
                    "category": "safe_compliance",
                    "rationale": str(compliance_judge.get("rationale", "")),
                },
                model=args.generator_model,
            )
            write_record(out_file, record)
            counts["safe_compliance"] += 1

        judged_model_outputs = 0
        for row in risk_rows:
            if judged_model_outputs >= args.max_judged_model_outputs:
                break
            prompt = str(row["user_input"])
            response = str(row["model_output"]).strip()
            try:
                output_judge = judge.score(prompt, response)
            except Exception:
                skipped["risk_judge"] += 1
                continue
            label, label_name = teacher_label_from_judge(output_judge)
            if label_name == "safe_compliance":
                # Risk prompts that got benign answers are less useful than refusal/unsafe contrast.
                skipped["risk_judge"] += 1
                continue
            record = toxicchat_source_record(
                row=row,
                response=response,
                label=label,
                label_name=label_name,
                response_kind="toxicchat_model_output",
                judge=output_judge,
                model="toxicchat_model_output",
            )
            write_record(out_file, record)
            counts[label_name] += 1
            judged_model_outputs += 1

    print(
        json.dumps(
            {
                "out_path": str(out_path),
                "dataset": "lmsys/toxic-chat",
                "dataset_config": args.dataset_config,
                "split": args.split,
                "selected_safe_candidates": len(safe_rows),
                "selected_risk_candidates": len(risk_rows),
                "counts": counts,
                "skipped": skipped,
                "max_safe_compliance": args.max_safe_compliance,
                "max_judged_model_outputs": args.max_judged_model_outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
