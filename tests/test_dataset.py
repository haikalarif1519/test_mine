"""Unit tests for lstm_forecaster.dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Locate testdata relative to this file
_REPO_ROOT = Path(__file__).parent.parent
_SAMPLE_CSV = _REPO_ROOT / "testdata" / "sample.csv"
_SAMPLE_JSONL = _REPO_ROOT / "testdata" / "sample.jsonl"

from lstm_forecaster.dataset import (
    FEATURE_COLS,
    LABEL_COL,
    CSVDatasetLoader,
    JSONLDatasetLoader,
    Normalizer,
    WindowedDataset,
    label_from_telemetry,
    load_data,
)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_csv_loading():
    """CSVDatasetLoader returns a non-empty DataFrame with the expected columns."""
    loader = CSVDatasetLoader(_SAMPLE_CSV)
    loader.load()
    df = loader.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    for col in FEATURE_COLS + [LABEL_COL]:
        assert col in df.columns, f"Missing column: {col}"


def test_jsonl_loading():
    """JSONLDatasetLoader returns a non-empty DataFrame with the expected columns."""
    loader = JSONLDatasetLoader(_SAMPLE_JSONL)
    loader.load()
    df = loader.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    for col in FEATURE_COLS + [LABEL_COL]:
        assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Windowing tests
# ---------------------------------------------------------------------------


def test_windowing():
    """WindowedDataset produces correctly shaped tensors."""
    df = load_data(_SAMPLE_CSV)

    window_size = 30
    horizon = 10
    ds = WindowedDataset(df, FEATURE_COLS, LABEL_COL, window_size=window_size, horizon=horizon)

    expected_len = max(0, len(df) - window_size - horizon + 1)
    assert len(ds) == expected_len

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (window_size, len(FEATURE_COLS))
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert int(y) in (0, 1, 2, 3)


def test_windowing_last_item():
    """Last item in the dataset has a valid label index."""
    df = load_data(_SAMPLE_CSV)
    window_size = 20
    horizon = 5
    ds = WindowedDataset(df, FEATURE_COLS, LABEL_COL, window_size=window_size, horizon=horizon)

    x, y = ds[len(ds) - 1]
    assert x.shape == (window_size, len(FEATURE_COLS))
    assert int(y) in (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------


def test_normalizer():
    """Fitted Normalizer transforms data to approximately zero mean, unit std."""
    df = load_data(_SAMPLE_CSV)
    norm = Normalizer()
    transformed = norm.fit_transform(df)

    for col in FEATURE_COLS:
        col_vals = transformed[col].values
        assert abs(col_vals.mean()) < 0.1, f"{col}: mean not near 0"
        assert abs(col_vals.std() - 1.0) < 0.1, f"{col}: std not near 1"


def test_normalizer_serialization():
    """Normalizer round-trips through to_dict / from_dict."""
    df = load_data(_SAMPLE_CSV)
    norm = Normalizer()
    norm.fit(df)

    d = norm.to_dict()
    norm2 = Normalizer.from_dict(d)

    for col in FEATURE_COLS:
        assert abs(norm._mean[col] - norm2._mean[col]) < 1e-9
        assert abs(norm._std[col] - norm2._std[col]) < 1e-9


# ---------------------------------------------------------------------------
# label_from_telemetry tests
# ---------------------------------------------------------------------------


def test_label_from_telemetry_stress():
    """High CPU → Stress (label 3)."""
    row = pd.Series({
        "cpu_utilization": 85.0,
        "power_draw_watts": 45.0,
        "keyboard_interrupt_rate": 0.0,
        "mouse_interrupt_rate": 0.0,
        "memory_utilization": 80.0,
        "disk_io_mbps": 10.0,
        "network_mbps": 2.0,
    })
    assert label_from_telemetry(row) == 3


def test_label_from_telemetry_typing():
    """Active keyboard with moderate CPU → Typing (label 2)."""
    row = pd.Series({
        "cpu_utilization": 25.0,
        "power_draw_watts": 15.0,
        "keyboard_interrupt_rate": 10.0,
        "mouse_interrupt_rate": 8.0,
        "memory_utilization": 40.0,
        "disk_io_mbps": 2.0,
        "network_mbps": 1.0,
    })
    assert label_from_telemetry(row) == 2


def test_label_from_telemetry_idle():
    """Very low CPU and power → Idle (label 0)."""
    row = pd.Series({
        "cpu_utilization": 3.0,
        "power_draw_watts": 3.0,
        "keyboard_interrupt_rate": 0.0,
        "mouse_interrupt_rate": 0.0,
        "memory_utilization": 10.0,
        "disk_io_mbps": 0.5,
        "network_mbps": 0.1,
    })
    assert label_from_telemetry(row) == 0


def test_label_from_telemetry_power_cycling():
    """Moderate activity, no keyboard → Power-cycling (label 1)."""
    row = pd.Series({
        "cpu_utilization": 15.0,
        "power_draw_watts": 12.0,
        "keyboard_interrupt_rate": 1.0,
        "mouse_interrupt_rate": 0.5,
        "memory_utilization": 30.0,
        "disk_io_mbps": 2.0,
        "network_mbps": 0.5,
    })
    assert label_from_telemetry(row) == 1
