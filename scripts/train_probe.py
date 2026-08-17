#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from activation_probe_mvp.activations import (
    extract_final_token_hidden_state,
    load_model_and_tokenizer,
)
from activation_probe_mvp.chat import format_exchange
from activation_probe_mvp.data import balanced_sample, load_jsonl
from activation_probe_mvp.probe_models import resolve_probe_model_route
from activation_probe_mvp.training import evaluate_probe, save_probe, train_linear_probe


def write_progress(path: str | None, payload: dict) -> None:
    if path is None:
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = progress_path.with_suffix(f"{progress_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                **payload,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    temporary_path.replace(progress_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="google/gemma-4-E2B-it",
        help="Official Gemma, Qwen, Kimi, or GLM text-generation checkpoint.",
    )
    parser.add_argument("--data_path", default="data/training_data.jsonl")
    parser.add_argument("--layer", type=int, default=-4)
    parser.add_argument("--out_dir", default="./probe_out_public_safety")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20000,
        help="Maximum examples to use after deterministic label-balanced sampling. Use 0 for all.",
    )
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--progress_path", default=None)
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable tokenizer/processor chat templates and use simple User/Assistant formatting.",
    )
    args = parser.parse_args()
    route = resolve_probe_model_route(args.model_id)

    write_progress(args.progress_path, {"phase": "loading_model", "percent": 0.0})
    print(f"Loading model: {args.model_id}")
    model, tokenizer_or_processor, device = load_model_and_tokenizer(args.model_id)

    all_examples = load_jsonl(args.data_path)
    examples = balanced_sample(
        all_examples,
        max_examples=args.max_examples if args.max_examples > 0 else None,
        seed=args.sample_seed,
    )
    invalid_labels = sorted({example.label for example in examples} - {0, 1})
    if invalid_labels:
        raise ValueError(f"Probe training requires binary labels 0 or 1; found {invalid_labels}")
    print(f"Loaded {len(all_examples)} examples; training on {len(examples)}")

    if len(examples) < 4:
        raise ValueError("Need at least a few examples. For real use, use thousands.")

    Xs = []
    ys = []

    use_chat_template = not args.no_chat_template

    print(f"Extracting activations from layer {args.layer} on {device}")
    print(f"Chat template enabled: {use_chat_template}")
    extraction_started_at = time.monotonic()
    progress_interval = max(len(examples) // 100, 1)
    write_progress(
        args.progress_path,
        {
            "phase": "extracting_activations",
            "examples_processed": 0,
            "total_examples": len(examples),
            "percent": 0.0,
            "eta_seconds": None,
        },
    )

    for index, ex in enumerate(tqdm(examples), start=1):
        text = format_exchange(
            tokenizer_or_processor=tokenizer_or_processor,
            prompt=ex.prompt,
            response=ex.response,
            use_chat_template=use_chat_template,
        )

        hidden = extract_final_token_hidden_state(
            model=model,
            tokenizer_or_processor=tokenizer_or_processor,
            text=text,
            layer=args.layer,
            device=device,
        )

        Xs.append(hidden.squeeze(0))
        ys.append(ex.label)
        if index % progress_interval == 0 or index == len(examples):
            elapsed_seconds = max(time.monotonic() - extraction_started_at, 0.0)
            examples_per_second = index / elapsed_seconds if elapsed_seconds > 0 else 0
            remaining_examples = len(examples) - index
            eta_seconds = (
                remaining_examples / examples_per_second if examples_per_second > 0 else None
            )
            write_progress(
                args.progress_path,
                {
                    "phase": "extracting_activations",
                    "examples_processed": index,
                    "total_examples": len(examples),
                    "percent": round(95 * index / len(examples), 2),
                    "elapsed_seconds": round(elapsed_seconds),
                    "eta_seconds": round(eta_seconds) if eta_seconds is not None else None,
                },
            )

    X = torch.stack(Xs)
    y = torch.tensor(ys, dtype=torch.float32)

    print(f"Training probe on X={tuple(X.shape)}")
    write_progress(
        args.progress_path,
        {
            "phase": "training_probe",
            "examples_processed": len(examples),
            "total_examples": len(examples),
            "percent": 95.0,
            "eta_seconds": None,
        },
    )

    probe = train_linear_probe(
        X=X,
        y=y,
        epochs=args.epochs,
        lr=args.lr,
    )

    metrics = evaluate_probe(probe, X, y)

    print("Metrics on training set:")
    print(json.dumps(metrics, indent=2))

    config = {
        "model_id": args.model_id,
        "model_family": route.family,
        "recommended_gpu": route.gpu,
        "estimated_parameters_b": route.estimated_parameters_b,
        "layer": args.layer,
        "hidden_size": int(X.shape[-1]),
        "threshold": 0.65,
        "pause_threshold": 0.75,
        "probe_every_n_tokens": 4,
        "smoothing_window": 4,
        "use_chat_template": use_chat_template,
        "metrics": metrics,
        "note": "MVP config. Calibrate thresholds on validation data before production use.",
    }

    save_probe(probe, args.out_dir, config)
    write_progress(
        args.progress_path,
        {
            "phase": "completed",
            "examples_processed": len(examples),
            "total_examples": len(examples),
            "percent": 100.0,
            "eta_seconds": 0,
        },
    )

    print(f"Saved probe to {args.out_dir}")


if __name__ == "__main__":
    main()
