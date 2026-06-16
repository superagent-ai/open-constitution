from __future__ import annotations

import torch

from activation_probe_mvp.modeling import LinearProbe, RiskSmoother


def test_risk_smoother_averages_recent_window():
    smoother = RiskSmoother(window_size=3)

    assert smoother.update(0.0) == 0.0
    assert smoother.update(0.6) == 0.3
    assert smoother.update(0.9) == 0.5
    assert smoother.update(0.3) == 0.6
    assert smoother.values == [0.6, 0.9, 0.3]


def test_linear_probe_returns_one_score_per_example():
    probe = LinearProbe(hidden_size=4)

    scores = probe(torch.ones(2, 4))

    assert scores.shape == (2,)
