from __future__ import annotations


def _get_chat_template_owner(tokenizer_or_processor):
    """
    Gemma 4 uses AutoProcessor, many text-only models use AutoTokenizer.
    Both may expose apply_chat_template directly, or the processor may expose
    a .tokenizer with apply_chat_template.
    """
    if hasattr(tokenizer_or_processor, "apply_chat_template"):
        return tokenizer_or_processor

    tok = getattr(tokenizer_or_processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "apply_chat_template"):
        return tok

    return None


def format_exchange(
    tokenizer_or_processor,
    prompt: str,
    response: str,
    use_chat_template: bool = True,
) -> str:
    """
    Format a complete prompt + assistant response for probe training.

    If the model/processor has a chat template, use it. Otherwise fall back to
    a simple User/Assistant format.
    """
    if use_chat_template:
        owner = _get_chat_template_owner(tokenizer_or_processor)

        if owner is not None:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

            try:
                return owner.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except TypeError:
                # Some processors/templates have slightly different signatures.
                return owner.apply_chat_template(messages, tokenize=False)

    return f"User:\n{prompt}\n\nAssistant:\n{response}"


def format_generation_prompt(
    tokenizer_or_processor,
    prompt: str,
    use_chat_template: bool = True,
) -> str:
    """
    Format a user prompt for generation.

    For chat models, add_generation_prompt=True leaves the assistant turn open.
    """
    if use_chat_template:
        owner = _get_chat_template_owner(tokenizer_or_processor)

        if owner is not None:
            messages = [
                {"role": "user", "content": prompt},
            ]

            try:
                return owner.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                return owner.apply_chat_template(messages, tokenize=False)

    return f"User:\n{prompt}\n\nAssistant:\n"
