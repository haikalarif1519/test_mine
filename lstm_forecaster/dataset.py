"""Dataset loading, labeling, normalization, and windowing utilities."""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    "cpu_utilization",
    "power_draw_watts",
    "keyboard_interrupt_rate",
    "mouse_interrupt_rate",
    "memory_utilization",
    "disk_io_mbps",
    "network_mbps",
]

LABEL_COL: str = "scenario_label"

SCENARIO_NAMES: List[str] = [
    "Idle",
    "Power-cycling",
    "Typing/low-power",
    "Stress/high-power",
]

# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------


def label_from_telemetry(row: pd.Series) -> int:
    """Derive a scenario label from a single telemetry row using threshold rules.

    Priority order (highest to lowest):
      3 — Stress/high-power  : cpu > 70 OR power > 30 W
      2 — Typing/low-power   : keyboard > 5 OR mouse > 5 interrupts/s
      0 — Idle               : cpu < 10 AND power < 10 W
      1 — Power-cycling risk : everything else
    """
    if row["cpu_utilization"] > 70 or row["power_draw_watts"] > 30:
        return 3
    if row["keyboard_interrupt_rate"] > 5 or row["mouse_interrupt_rate"] > 5:
        return 2
    if row["cpu_utilization"] < 10 and row["power_draw_watts"] < 10:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


class BaseDatasetLoader(abc.ABC):
    """Abstract base class for dataset loaders."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load data from the source into memory."""

    @abc.abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """Return the loaded data as a pandas DataFrame."""


class CSVDatasetLoader(BaseDatasetLoader):
    """Loads telemetry data from a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Parse the CSV file into a DataFrame."""
        self._df = pd.read_csv(self._path)

    def get_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df.copy()


class JSONLDatasetLoader(BaseDatasetLoader):
    """Loads telemetry data from a JSONL (newline-delimited JSON) file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Parse the JSONL file into a DataFrame."""
        records = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        self._df = pd.DataFrame(records)

    def get_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df.copy()


def load_data(path: str | Path) -> pd.DataFrame:
    """Auto-detect file format and return a DataFrame.

    Supports ``.csv`` and ``.jsonl`` extensions.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        loader: BaseDatasetLoader = CSVDatasetLoader(path)
    elif suffix == ".jsonl":
        loader = JSONLDatasetLoader(path)
    else:
        raise ValueError(f"Unsupported file extension '{suffix}'. Use .csv or .jsonl.")
    return loader.get_dataframe()


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class Normalizer:
    """Z-score normalizer fitted on training data.

    Parameters are stored as plain dicts so they can be serialized to JSON
    alongside the model checkpoint.
    """

    _EPS: float = 1e-8

    def __init__(self) -> None:
        self._mean: Dict[str, float] = {}
        self._std: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fitting / transforming
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> "Normalizer":
        """Compute per-column mean and std from *df*."""
        cols = cols or FEATURE_COLS
        for col in cols:
            self._mean[col] = float(df[col].mean())
            self._std[col] = float(df[col].std(ddof=0))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization using fitted statistics."""
        df = df.copy()
        for col, mean in self._mean.items():
            if col in df.columns:
                df[col] = (df[col] - mean) / (self._std[col] + self._EPS)
        return df

    def fit_transform(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit on *df* then transform and return it."""
        self.fit(df, cols)
        return self.transform(df)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize normalizer parameters to a plain dict."""
        return {"mean": dict(self._mean), "std": dict(self._std)}

    @classmethod
    def from_dict(cls, d: Dict) -> "Normalizer":
        """Reconstruct a Normalizer from a serialized dict."""
        obj = cls()
        obj._mean = d["mean"]
        obj._std = d["std"]
        return obj


# ---------------------------------------------------------------------------
# Windowed Dataset
# ---------------------------------------------------------------------------


class WindowedDataset(Dataset):
    """Sliding-window dataset for sequence-to-label LSTM training.

    For each index *i* the sample is:
      - **window**  : rows [i, i + window_size)          — shape (window_size, len(feature_cols))
      - **label**   : scenario at row i + window_size + horizon - 1

    ``__len__`` returns ``max(0, len(df) - window_size - horizon + 1)``.

    If *label_col* is absent from *df*, labels are derived via
    :func:`label_from_telemetry`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = FEATURE_COLS,
        label_col: str = LABEL_COL,
        window_size: int = 120,
        horizon: int = 60,
        stride: int = 1,
    ) -> None:
        self._window_size = window_size
        self._horizon = horizon
        self._stride = stride
        self._feature_cols = feature_cols

        # Ensure we have a label column.
        if label_col not in df.columns:
            df = df.copy()
            df[label_col] = df.apply(label_from_telemetry, axis=1)

        self._features: np.ndarray = df[feature_cols].to_numpy(dtype=np.float32)
        self._labels: np.ndarray = df[label_col].to_numpy(dtype=np.int64)

        # Indices of valid window starts.
        max_start = len(df) - window_size - horizon + 1
        self._indices = list(range(0, max(0, max_start), stride))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        start = self._indices[idx]
        window = self._features[start : start + self._window_size]  # (window_size, F)
        label_idx = start + self._window_size + self._horizon - 1
        label = self._labels[label_idx]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
