from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class Artifacts:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float

class TinyNet(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def _pick_device(device: str) -> torch.device:
    device = device.lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    # auto
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

def train_model(df: pd.DataFrame, feature_cols: list[str], out_dir: Path, device: str = "auto"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    X = df[feature_cols].astype(float).to_numpy()
    y = df["choice"].astype(int).to_numpy()  # 1=Uber, 0=TTC

    Xs, mean, std = _standardize(X)

    dev = _pick_device(device)
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else None
    
    # Print device info for debugging
    if dev.type == "cuda":
        print(f"Using GPU: {gpu_name}")
    else:
        print(f"CUDA not available. Using CPU (torch.cuda.is_available()={torch.cuda.is_available()})")

    # Model
    torch.manual_seed(42)
    model = TinyNet(d_in=Xs.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(Xs, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev)

    # Train (tiny, fast)
    model.train()
    for epoch in range(200):
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()

    # Save model weights + preprocessing stats
    ckpt = {
        "state_dict": model.state_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_cols": feature_cols,
    }
    torch.save(ckpt, out_dir / "model.pt")

    # Save run metadata (this is your “trained on 5090 locally” proof)
    meta = {
        "device": str(dev),
        "gpu_name": gpu_name,
        "num_rows": int(len(df)),
        "epochs": 200,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))


def predict_choice(x_row: dict, feature_cols: list[str], model_dir: Path) -> str:
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32) + 1e-8

    # Load model
    model = TinyNet(d_in=len(feature_cols))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = np.array([float(x_row[c]) for c in feature_cols], dtype=np.float32)
    xs = (x - mean) / std
    with torch.no_grad():
        logit = model(torch.tensor(xs).unsqueeze(0)).item()
        prob_uber = 1 / (1 + np.exp(-logit))

    recommendation = "UBER" if prob_uber >= 0.5 else "TTC"
    conf = prob_uber if recommendation == "UBER" else (1 - prob_uber)

    # Simple “reasons”: show biggest standardized feature magnitudes (honest + quick)
    top_idx = np.argsort(np.abs(xs))[::-1][:3]
    reasons = [f"- {feature_cols[i]} is high/low vs your usual" for i in top_idx]

    return (
        f"Recommendation: {recommendation}  (confidence ~ {conf*100:.0f}%)\n"
        "Top drivers (rough):\n" + "\n".join(reasons)
    )