from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_gemma4(model_id: str) -> bool:
    return "gemma-4" in model_id.lower()


def _set_pad_token(tokenizer_or_processor) -> None:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)

    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_model_and_tokenizer(model_id: str, device: str | None = None):
    """
    Load a model + tokenizer/processor.

    Standard text-only models use:
      AutoTokenizer + AutoModelForCausalLM

    Gemma 4 uses:
      AutoProcessor + AutoModelForImageTextToText

    The returned second object may therefore be a tokenizer or a processor.
    The rest of the repo treats it as tokenizer_or_processor.
    """
    device = device or get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    if _is_gemma4(model_id):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _set_pad_token(processor)

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
            device_map=None,
        )

        model.to(device)
        model.eval()

        return model, processor, device

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _set_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def tokenize_text(tokenizer_or_processor, text: str, device: str, max_length: int | None = None):
    kwargs = {
        "return_tensors": "pt",
    }

    if max_length is not None:
        kwargs.update(
            {
                "truncation": True,
                "max_length": max_length,
            }
        )

    try:
        inputs = tokenizer_or_processor(text=text, **kwargs)
    except TypeError:
        inputs = tokenizer_or_processor(text, **kwargs)

    return inputs.to(device)


def get_selected_hidden_state(outputs, layer: int) -> torch.Tensor:
    """
    Return hidden state tensor for the selected layer.

    Most causal LMs expose outputs.hidden_states.
    Encoder/decoder or image-text-to-text models may expose decoder_hidden_states.
    """
    hidden_states = getattr(outputs, "hidden_states", None)

    if hidden_states is None:
        hidden_states = getattr(outputs, "decoder_hidden_states", None)

    if hidden_states is None:
        raise RuntimeError(
            "Model output did not include hidden_states or decoder_hidden_states. "
            "This model may not support output_hidden_states=True through this API."
        )

    return hidden_states[layer]


@torch.no_grad()
def extract_final_token_hidden_state(
    model,
    tokenizer_or_processor,
    text: str,
    layer: int,
    device: str,
    max_length: int = 2048,
) -> torch.Tensor:
    inputs = tokenize_text(
        tokenizer_or_processor=tokenizer_or_processor,
        text=text,
        device=device,
        max_length=max_length,
    )

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    selected = get_selected_hidden_state(outputs, layer)  # [batch, seq, hidden]
    final_token = selected[:, -1, :]  # [batch, hidden]

    return final_token.float().detach().cpu()
