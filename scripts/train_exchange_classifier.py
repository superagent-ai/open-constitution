#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from activation_probe_mvp.exchange_training import (
    DEFAULT_ALLOW_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_CLASSIFIER_MODEL_ID,
    DEFAULT_TRAINING_DATA_PATH,
    train_exchange_classifier,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_CLASSIFIER_MODEL_ID)
    parser.add_argument("--data_path", default=DEFAULT_TRAINING_DATA_PATH)
    parser.add_argument("--out_dir", "--output-dir", dest="out_dir", default="./classifier_out")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix_augment", action="store_true")
    parser.add_argument("--prefix_copies", type=int, default=1)
    parser.add_argument("--allow_threshold", type=float, default=DEFAULT_ALLOW_THRESHOLD)
    parser.add_argument("--block_threshold", type=float, default=DEFAULT_BLOCK_THRESHOLD)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--disable_tqdm", action="store_true", default=True)
    parser.add_argument("--show_tqdm", action="store_false", dest="disable_tqdm")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    metrics = train_exchange_classifier(
        model_id=args.model_id,
        data_path=args.data_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        validation_size=args.validation_size,
        seed=args.seed,
        prefix_augment=args.prefix_augment,
        prefix_copies=args.prefix_copies,
        allow_threshold=args.allow_threshold,
        block_threshold=args.block_threshold,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        disable_tqdm=args.disable_tqdm,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
