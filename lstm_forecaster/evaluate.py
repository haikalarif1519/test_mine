"""Evaluation utilities for the LSTM power-management forecaster."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lstm_forecaster.dataset import (
    FEATURE_COLS,
    LABEL_COL,
    SCENARIO_NAMES,
    Normalizer,
    WindowedDataset,
    label_from_telemetry,
    load_data,
)
from lstm_forecaster.model import LSTMForecaster


def load_checkpoint(checkpoint_path: str | Path) -> Dict:
    """Load a training checkpoint from disk.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.

    Returns:
        The checkpoint dict containing ``state_dict``, ``config``,
        ``normalizer``, and ``feature_cols``.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def evaluate(
    checkpoint_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
) -> Dict:
    """Load a checkpoint, run inference, and compute evaluation metrics.

    Metrics computed:
      - accuracy
      - macro F1 (and per-class F1)
      - confusion matrix
      - Brier score

    Args:
        checkpoint_path: Path to saved ``.pt`` checkpoint.
        data_path:       Path to the raw telemetry CSV or JSONL file.
        output_dir:      Directory where ``metrics.json`` will be written.
        split:           Which portion of the data to evaluate on.
                         ``"test"`` uses the last 15 %, ``"val"`` uses the
                         middle 15 %, ``"all"`` uses the full file.

    Returns:
        Dict containing all computed metrics.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint["config"]
    feature_cols = checkpoint.get("feature_cols", FEATURE_COLS)

    # ------------------------------------------------------------------ #
    # Reconstruct normalizer and model
    # ------------------------------------------------------------------ #
    normalizer = Normalizer.from_dict(checkpoint["normalizer"])

    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        num_classes=cfg.get("num_classes", 4),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ------------------------------------------------------------------ #
    # Load and select the requested split
    # ------------------------------------------------------------------ #
    df = load_data(data_path)

    if LABEL_COL not in df.columns:
        df[LABEL_COL] = df.apply(label_from_telemetry, axis=1)

    n = len(df)
    val_ratio: float = cfg.get("val_ratio", 0.15)
    test_ratio: float = cfg.get("test_ratio", 0.15)

    if split == "train":
        train_end = int(n * (1.0 - val_ratio - test_ratio))
        eval_df = df.iloc[:train_end].reset_index(drop=True)
    elif split == "val":
        train_end = int(n * (1.0 - val_ratio - test_ratio))
        val_end = int(n * (1.0 - test_ratio))
        eval_df = df.iloc[train_end:val_end].reset_index(drop=True)
    elif split == "test":
        val_end = int(n * (1.0 - test_ratio))
        eval_df = df.iloc[val_end:].reset_index(drop=True)
    else:  # "all"
        eval_df = df.reset_index(drop=True)

    eval_df = normalizer.transform(eval_df)

    window_size: int = cfg.get("window_size", 120)
    horizon: int = cfg.get("horizon", 60)
    stride: int = cfg.get("stride", 1)

    dataset = WindowedDataset(eval_df, feature_cols, LABEL_COL, window_size, horizon, stride)

    if len(dataset) == 0:
        raise ValueError(
            f"Evaluation dataset for split='{split}' is empty. "
            "The split may be too short for the chosen window_size and horizon."
        )

    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 64), shuffle=False)

    # ------------------------------------------------------------------ #
    # Collect predictions
    # ------------------------------------------------------------------ #
    all_labels: list = []
    all_probs: list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y.numpy().tolist())

    labels = np.array(all_labels, dtype=np.int64)
    probs = np.concatenate(all_probs, axis=0)  # (N, num_classes)
    preds = probs.argmax(axis=1)

    num_classes = cfg.get("num_classes", 4)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    accuracy = float((preds == labels).mean())

    # Per-class F1 (manual, avoids sklearn dependency in this module)
    f1_per_class = _compute_f1_per_class(labels, preds, num_classes)
    macro_f1 = float(np.mean(f1_per_class))

    conf_matrix = _compute_confusion_matrix(labels, preds, num_classes)

    # Brier score: mean squared error of probabilities vs one-hot
    one_hot = np.eye(num_classes)[labels]
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    metrics: Dict = {
        "split": split,
        "num_samples": int(len(labels)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "brier_score": brier_score,
        "per_class_f1": {SCENARIO_NAMES[i]: float(f1_per_class[i]) for i in range(num_classes)},
        "confusion_matrix": conf_matrix.tolist(),
    }

    # ------------------------------------------------------------------ #
    # Save metrics
    # ------------------------------------------------------------------ #
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    print(f"\n=== Evaluation ({split}) ===")
    print(f"  Samples   : {metrics['num_samples']}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print(f"  Brier     : {brier_score:.4f}")
    print("  Per-class F1:")
    for name, f1 in metrics["per_class_f1"].items():
        print(f"    {name:<22}: {f1:.4f}")
    print(f"  Metrics saved to: {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_f1_per_class(
    labels: np.ndarray, preds: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute per-class F1 scores."""
    f1 = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1[c] = 2.0 * precision * recall / (precision + recall)
    return f1


def _compute_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute an (num_classes × num_classes) confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, preds):
        cm[int(true), int(pred)] += 1
    return cm
