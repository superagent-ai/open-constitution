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
from activation_probe_mvp.training import evaluate_probe, train_linear_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it")
    parser.add_argument("--data_path", default="data/examples.jsonl")
    parser.add_argument("--layers", default="-2,-4,-6,-8,-10,-12")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable tokenizer/processor chat templates and use simple User/Assistant formatting.",
    )
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]
    examples = load_jsonl(args.data_path)

    model, tokenizer_or_processor, device = load_model_and_tokenizer(args.model_id)

    results = {}
    use_chat_template = not args.no_chat_template

    print(f"Chat template enabled: {use_chat_template}")

    for layer in layers:
        Xs = []
        ys = []

        print(f"\nLayer {layer}")

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
                layer=layer,
                device=device,
            )

            Xs.append(hidden.squeeze(0))
            ys.append(ex.label)

        X = torch.stack(Xs)
        y = torch.tensor(ys, dtype=torch.float32)

        probe = train_linear_probe(X, y, epochs=args.epochs)
        metrics = evaluate_probe(probe, X, y)

        results[str(layer)] = metrics

        print(json.dumps(metrics, indent=2))

    print("\n--- sweep results ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
