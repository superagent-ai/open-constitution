from __future__ import annotations

from .exchange_classifier import ExchangeDecision, decision_from_score


def classify_exchange_stub(prompt: str, partial_response: str) -> ExchangeDecision:
    """
    Placeholder for a real second-stage exchange classifier.

    Replace this with:
    - a fine-tuned Superagent Guard classifier,
    - a larger local model,
    - or an external moderation/exchange classifier.

    The point: the activation probe should usually escalate to this, not be the
    sole safety decision-maker.
    """
    suspicious_keywords = [
        "steal",
        "exfiltrate",
        "credential",
        "malware",
        "persistence",
        "phishing",
        "bypass",
    ]

    text = f"{prompt}\n{partial_response}".lower()
    hits = sum(1 for k in suspicious_keywords if k in text)
    score = min(1.0, hits / 2.0)

    return decision_from_score(
        score=score,
        reason="keyword_stub_only_do_not_use_in_production",
    )
