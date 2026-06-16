from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .modeling import LinearProbe


def train_linear_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    weight_decay: float = 0.01,
) -> LinearProbe:
    hidden_size = X.shape[-1]
    probe = LinearProbe(hidden_size)

    dataset = TensorDataset(X.float(), y.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    probe.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"epoch={epoch+1:03d} loss={total / max(len(loader), 1):.4f}")

    return probe


@torch.no_grad()
def evaluate_probe(probe: LinearProbe, X: torch.Tensor, y: torch.Tensor) -> dict:
    probe.eval()
    logits = probe(X.float())
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    yt = y.cpu().numpy().astype(int)

    metrics = {
        "accuracy": float(accuracy_score(yt, preds)),
        "precision": float(precision_score(yt, preds, zero_division=0)),
        "recall": float(recall_score(yt, preds, zero_division=0)),
        "f1": float(f1_score(yt, preds, zero_division=0)),
    }

    if len(set(yt.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(yt, probs))

    return metrics


def save_probe(probe: LinearProbe, out_dir: str | Path, config: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), out / "probe.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_probe(probe_path: str | Path, hidden_size: int) -> LinearProbe:
    probe = LinearProbe(hidden_size)
    state = torch.load(probe_path, map_location="cpu")
    probe.load_state_dict(state)
    probe.eval()
    return probe
