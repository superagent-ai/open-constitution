#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm

from activation_probe_mvp.activations import (
    extract_final_token_hidden_state,
    load_model_and_tokenizer,
)
from activation_probe_mvp.chat import format_exchange
from activation_probe_mvp.data import load_jsonl
from activation_probe_mvp.training import evaluate_probe, save_probe, train_linear_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="google/gemma-4-E2B-it",
        help="Default is Gemma 4 E2B instruction-tuned. Also works with Qwen/Llama-style causal LMs.",
    )
    parser.add_argument("--data_path", default="data/examples.jsonl")
    parser.add_argument("--layer", type=int, default=-4)
    parser.add_argument("--out_dir", default="./probe_out")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable tokenizer/processor chat templates and use simple User/Assistant formatting.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    model, tokenizer_or_processor, device = load_model_and_tokenizer(args.model_id)

    examples = load_jsonl(args.data_path)

    if len(examples) < 4:
        raise ValueError("Need at least a few examples. For real use, use thousands.")

    Xs = []
    ys = []

    use_chat_template = not args.no_chat_template

    print(f"Extracting activations from layer {args.layer} on {device}")
    print(f"Chat template enabled: {use_chat_template}")

    for ex in tqdm(examples):
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

    X = torch.stack(Xs)
    y = torch.tensor(ys, dtype=torch.float32)

    print(f"Training probe on X={tuple(X.shape)}")

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
        "layer": args.layer,
        "hidden_size": int(X.shape[-1]),
        "threshold": 0.65,
        "pause_threshold": 0.75,
        "probe_every_n_tokens": 4,
        "smoothing_window": 4,
        "use_chat_template": use_chat_template,
        "note": "MVP config. Calibrate thresholds on validation data before production use.",
    }

    save_probe(probe, args.out_dir, config)

    print(f"Saved probe to {args.out_dir}")


if __name__ == "__main__":
    main()
