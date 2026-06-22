#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from activation_probe_mvp.activations import (
    get_selected_hidden_state,
    load_model_and_tokenizer,
    tokenize_text,
)
from activation_probe_mvp.chat import format_generation_prompt
from activation_probe_mvp.exchange_classifier import DEFAULT_REFUSAL, ExchangeClassifier
from activation_probe_mvp.modeling import LinearProbe, RiskSmoother


def decode_token(tokenizer_or_processor, token: torch.Tensor) -> str:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    return tokenizer.decode(token.squeeze(0), skip_special_tokens=True)


@torch.no_grad()
def guarded_generate(
    model,
    tokenizer_or_processor,
    device: str,
    probe: LinearProbe,
    layer: int,
    prompt: str,
    max_new_tokens: int,
    probe_every_n_tokens: int,
    threshold: float,
    pause_threshold: float,
    smoothing_window: int,
    use_chat_template: bool,
    classifier: ExchangeClassifier | None = None,
    refusal_message: str = DEFAULT_REFUSAL,
):
    input_text = format_generation_prompt(
        tokenizer_or_processor=tokenizer_or_processor,
        prompt=prompt,
        use_chat_template=use_chat_template,
    )

    inputs = tokenize_text(
        tokenizer_or_processor=tokenizer_or_processor,
        text=input_text,
        device=device,
    )
    input_ids = inputs["input_ids"]

    smoother = RiskSmoother(window_size=smoothing_window)
    generated_tokens = []
    pending_tokens = []

    def flush_pending_tokens() -> None:
        if pending_tokens:
            print("".join(pending_tokens), end="", flush=True)
            pending_tokens.clear()

    print("\n--- guarded generation ---\n")

    for step in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        token_text = decode_token(tokenizer_or_processor, next_token)
        generated_tokens.append(token_text)
        pending_tokens.append(token_text)

        should_probe = (step + 1) % probe_every_n_tokens == 0

        if not should_probe:
            continue

        selected = get_selected_hidden_state(outputs, layer)
        hidden = selected[:, -1, :].float().cpu()

        risk = torch.sigmoid(probe(hidden)).item()
        smoothed = smoother.update(risk)

        if smoothed >= pause_threshold:
            partial = "".join(generated_tokens)

            print(f"\n[probe] step={step + 1} risk={risk:.3f} smoothed={smoothed:.3f}")
            print("[guard] PAUSE: high risk from activation probe.")

            if classifier is None:
                print("[guard] No exchange classifier configured; returning paused state.")
                print("\n--- partial response ---")
                print(partial)

                return {
                    "status": "paused",
                    "risk": risk,
                    "smoothed_risk": smoothed,
                    "partial_response": partial,
                }

            decision = classifier.classify(prompt, partial)
            print(
                "[classifier] "
                f"verdict={decision.verdict} score={decision.score:.3f} "
                f"allow_threshold={decision.allow_threshold:.3f} "
                f"block_threshold={decision.block_threshold:.3f}"
            )

            if decision.verdict == "allow":
                flush_pending_tokens()
                continue

            if decision.verdict == "uncertain":
                print("[guard] UNCERTAIN: holding buffered tokens for tighter monitoring.")
                return {
                    "status": "uncertain",
                    "risk": risk,
                    "smoothed_risk": smoothed,
                    "classifier_score": decision.score,
                    "classifier_verdict": decision.verdict,
                    "held_response_chars": len(partial),
                }

            print("\n--- refusal ---")
            print(refusal_message)
            return {
                "status": "blocked",
                "risk": risk,
                "smoothed_risk": smoothed,
                "classifier_score": decision.score,
                "classifier_verdict": decision.verdict,
                "response": refusal_message,
                "held_response_chars": len(partial),
            }

        flush_pending_tokens()
        print(f"\n[probe] step={step + 1} risk={risk:.3f} smoothed={smoothed:.3f}")

        if smoothed >= threshold:
            print("[guard] suspicious: would escalate soon if risk persists.")

    flush_pending_tokens()
    final = "".join(generated_tokens)

    print("\n\n[guard] completed without pause.")

    return {
        "status": "completed",
        "response": final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--classifier_dir", default=None)
    parser.add_argument("--classifier_allow_threshold", type=float, default=None)
    parser.add_argument("--classifier_block_threshold", type=float, default=None)
    parser.add_argument(
        "--skip_classifier",
        action="store_true",
        help="Keep probe-only pause behavior even when a classifier directory is available.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable tokenizer/processor chat templates and use simple User/Assistant formatting.",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
    model_id = args.model_id or config["model_id"]

    use_chat_template = bool(config.get("use_chat_template", True))
    if args.no_chat_template:
        use_chat_template = False

    model, tokenizer_or_processor, device = load_model_and_tokenizer(model_id)

    probe = LinearProbe(config["hidden_size"])
    state = torch.load(args.probe_path, map_location="cpu")
    probe.load_state_dict(state)
    probe.eval()

    classifier = None
    if args.classifier_dir is not None and not args.skip_classifier:
        classifier = ExchangeClassifier.from_pretrained(
            args.classifier_dir,
            allow_threshold=args.classifier_allow_threshold,
            block_threshold=args.classifier_block_threshold,
        )

    result = guarded_generate(
        model=model,
        tokenizer_or_processor=tokenizer_or_processor,
        device=device,
        probe=probe,
        layer=int(config["layer"]),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        probe_every_n_tokens=int(config["probe_every_n_tokens"]),
        threshold=float(config["threshold"]),
        pause_threshold=float(config["pause_threshold"]),
        smoothing_window=int(config["smoothing_window"]),
        use_chat_template=use_chat_template,
        classifier=classifier,
    )

    print("\n\n--- result ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
