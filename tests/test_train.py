"""Smoke test for the LSTM training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lstm_forecaster.dataset import FEATURE_COLS, LABEL_COL
from lstm_forecaster.train import train


def _make_synthetic_df(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """Create a small synthetic DataFrame covering all 4 scenarios."""
    rng = np.random.default_rng(seed)
    rows_per_seg = n_rows // 4

    segments = []
    # Idle
    idle = pd.DataFrame({
        "cpu_utilization":         rng.uniform(2, 8, rows_per_seg),
        "power_draw_watts":        rng.uniform(2, 8, rows_per_seg),
        "keyboard_interrupt_rate": rng.uniform(0, 0.5, rows_per_seg),
        "mouse_interrupt_rate":    rng.uniform(0, 0.5, rows_per_seg),
        "memory_utilization":      rng.uniform(15, 25, rows_per_seg),
        "disk_io_mbps":            rng.uniform(0.5, 1.5, rows_per_seg),
        "network_mbps":            rng.uniform(0.1, 0.5, rows_per_seg),
        LABEL_COL:                 0,
    })
    segments.append(idle)

    # Typing
    typing = pd.DataFrame({
        "cpu_utilization":         rng.uniform(20, 30, rows_per_seg),
        "power_draw_watts":        rng.uniform(12, 18, rows_per_seg),
        "keyboard_interrupt_rate": rng.uniform(8, 12, rows_per_seg),
        "mouse_interrupt_rate":    rng.uniform(6, 10, rows_per_seg),
        "memory_utilization":      rng.uniform(35, 45, rows_per_seg),
        "disk_io_mbps":            rng.uniform(1.5, 2.5, rows_per_seg),
        "network_mbps":            rng.uniform(0.8, 1.2, rows_per_seg),
        LABEL_COL:                 2,
    })
    segments.append(typing)

    # Stress
    stress = pd.DataFrame({
        "cpu_utilization":         rng.uniform(80, 92, rows_per_seg),
        "power_draw_watts":        rng.uniform(40, 50, rows_per_seg),
        "keyboard_interrupt_rate": rng.uniform(0, 2, rows_per_seg),
        "mouse_interrupt_rate":    rng.uniform(0, 1, rows_per_seg),
        "memory_utilization":      rng.uniform(75, 85, rows_per_seg),
        "disk_io_mbps":            rng.uniform(18, 22, rows_per_seg),
        "network_mbps":            rng.uniform(4, 6, rows_per_seg),
        LABEL_COL:                 3,
    })
    segments.append(stress)

    # Power-cycling
    cycling = pd.DataFrame({
        "cpu_utilization":         rng.uniform(12, 18, rows_per_seg),
        "power_draw_watts":        rng.uniform(10, 14, rows_per_seg),
        "keyboard_interrupt_rate": rng.uniform(1, 3, rows_per_seg),
        "mouse_interrupt_rate":    rng.uniform(0.5, 1.5, rows_per_seg),
        "memory_utilization":      rng.uniform(25, 35, rows_per_seg),
        "disk_io_mbps":            rng.uniform(2.5, 3.5, rows_per_seg),
        "network_mbps":            rng.uniform(0.8, 1.2, rows_per_seg),
        LABEL_COL:                 1,
    })
    segments.append(cycling)

    df = pd.concat(segments, ignore_index=True)
    return df


def test_smoke_train(tmp_path):
    """Train for 1 epoch on tiny synthetic data and verify checkpoint is created."""
    df = _make_synthetic_df(n_rows=300, seed=42)

    # Save to a temp CSV so train() can load it
    data_path = tmp_path / "smoke_data.csv"
    df.to_csv(data_path, index=False)

    config = {
        "data_path": str(data_path),
        "window_size": 10,
        "horizon": 5,
        "stride": 1,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "lr": 1e-3,
        "batch_size": 16,
        "max_epochs": 1,
        "patience": 5,
        "seed": 42,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "use_class_weights": False,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "num_classes": 4,
    }

    ckpt_path = train(config)

    assert Path(ckpt_path).exists(), f"Checkpoint not found at {ckpt_path}"

    # Verify checkpoint contents
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "state_dict" in checkpoint
    assert "config" in checkpoint
    assert "normalizer" in checkpoint
    assert "feature_cols" in checkpoint

    # Verify normalizer has expected keys
    norm_dict = checkpoint["normalizer"]
    assert "mean" in norm_dict
    assert "std" in norm_dict
