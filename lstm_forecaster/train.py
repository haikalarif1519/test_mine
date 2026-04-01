"""Training loop for the LSTM power-management scenario forecaster."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lstm_forecaster.dataset import (
    FEATURE_COLS,
    LABEL_COL,
    Normalizer,
    WindowedDataset,
    label_from_telemetry,
    load_data,
)
from lstm_forecaster.model import LSTMForecaster

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    "data_path": "testdata/sample.csv",
    "window_size": 120,
    "horizon": 60,
    "stride": 1,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 5,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "use_class_weights": True,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "num_classes": 4,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    Args:
        labels:      1-D integer array of class labels.
        num_classes: Total number of classes.

    Returns:
        Float tensor of shape ``(num_classes,)`` with weights proportional
        to ``1 / class_frequency``, normalized so the minimum weight is 1.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.min()  # normalize
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Per-epoch routines
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run a single training epoch.

    Returns:
        Tuple of (average loss, accuracy) over the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a validation/test loader.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(config: Dict) -> Path:
    """Run the full training pipeline.

    Args:
        config: Configuration dict. See :data:`DEFAULT_CONFIG` for all keys.

    Returns:
        Path to the saved best-model checkpoint.
    """
    cfg = {**DEFAULT_CONFIG, **config}

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # 1. Load & label data
    # ------------------------------------------------------------------ #
    df = load_data(cfg["data_path"])

    if LABEL_COL not in df.columns:
        df[LABEL_COL] = df.apply(label_from_telemetry, axis=1)

    # ------------------------------------------------------------------ #
    # 2. Time-based split (70 / 15 / 15)
    # ------------------------------------------------------------------ #
    n = len(df)
    val_ratio: float = cfg["val_ratio"]
    test_ratio: float = cfg["test_ratio"]
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end = int(n * (1.0 - test_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 3. Normalize (fit on train only)
    # ------------------------------------------------------------------ #
    normalizer = Normalizer()
    train_df = normalizer.fit_transform(train_df)
    val_df = normalizer.transform(val_df)

    # ------------------------------------------------------------------ #
    # 4. Create windowed datasets
    # ------------------------------------------------------------------ #
    window_size: int = cfg["window_size"]
    horizon: int = cfg["horizon"]
    stride: int = cfg["stride"]

    train_ds = WindowedDataset(train_df, FEATURE_COLS, LABEL_COL, window_size, horizon, stride)
    val_ds = WindowedDataset(val_df, FEATURE_COLS, LABEL_COL, window_size, horizon, stride)

    if len(train_ds) == 0:
        raise ValueError(
            "Training dataset is empty. The data may be too short for the chosen "
            f"window_size={window_size} and horizon={horizon}."
        )

    batch_size: int = cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------ #
    # 5. Class weights
    # ------------------------------------------------------------------ #
    num_classes: int = cfg["num_classes"]
    weight_tensor: torch.Tensor | None = None
    if cfg["use_class_weights"]:
        all_train_labels: List[int] = [int(train_ds[i][1]) for i in range(len(train_ds))]
        weight_tensor = compute_class_weights(
            np.array(all_train_labels), num_classes=num_classes
        ).to(device)

    # ------------------------------------------------------------------ #
    # 6. Build model
    # ------------------------------------------------------------------ #
    model = LSTMForecaster(
        input_size=len(FEATURE_COLS),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_classes=num_classes,
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"]
    )

    # ------------------------------------------------------------------ #
    # 7. Training loop with early stopping
    # ------------------------------------------------------------------ #
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best_model.pt"

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{cfg['max_epochs']} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "normalizer": normalizer.to_dict(),
                    "feature_cols": FEATURE_COLS,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"Best checkpoint saved to: {best_ckpt_path}")
    return best_ckpt_path
