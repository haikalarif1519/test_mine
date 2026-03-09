"""
Data Preprocessing and Labelling.

Loads raw system metrics (from PostgreSQL or a CSV/DataFrame), applies
threshold-based labelling (Idle / Low Load / High Load), scales features,
and creates sequences suitable for LSTM training.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "cpu_utilization",
    "memory_utilization",
    "disk_read",
    "disk_write",
    "network_received",
    "network_sent",
    "system_idle_time",
]


# ── Labelling ───────────────────────────────────────────────────────────────

def _classify_metric(value, thresholds, invert=False):
    """
    Classify a single metric value.

    Parameters
    ----------
    value : float
        The metric value to classify.
    thresholds : dict
        Must contain ``"idle"`` and ``"low_load"`` keys with numeric values.
    invert : bool
        When *True* the comparison is inverted (used for system_idle_time
        where a **higher** value means the system is *more* idle).

    Returns
    -------
    str
        ``"idle"``, ``"low_load"`` or ``"high_load"``.
    """
    idle_thresh = thresholds["idle"]
    low_thresh = thresholds["low_load"]

    if invert:
        # Higher value → more idle (e.g. system idle %)
        if value >= idle_thresh:
            return "idle"
        if value >= low_thresh:
            return "low_load"
        return "high_load"

    # Normal: lower value → more idle
    if value <= idle_thresh:
        return "idle"
    if value <= low_thresh:
        return "low_load"
    return "high_load"


def label_row(row):
    """
    Label a single data row based on all metrics.

    The final label is the *highest* activity state detected across all
    metrics (i.e. if any metric indicates high_load, the row is labelled
    high_load).
    """
    priority = {"idle": 0, "low_load": 1, "high_load": 2}
    worst = "idle"

    for metric, thresholds in config.THRESHOLDS.items():
        value = row.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        invert = metric == "system_idle_time"
        state = _classify_metric(value, thresholds, invert=invert)
        if priority[state] > priority[worst]:
            worst = state

    return worst


def label_dataframe(df):
    """Add a ``state_label`` column to *df* (in-place) and return it."""
    df["state_label"] = df.apply(label_row, axis=1)
    return df


# ── Feature scaling ─────────────────────────────────────────────────────────

def fit_scaler(df):
    """Fit a MinMaxScaler on the feature columns and return it."""
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURE_COLUMNS].values)
    return scaler


def scale_features(df, scaler):
    """Return a copy of *df* with features scaled using *scaler*."""
    df = df.copy()
    df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS].values)
    return df


# ── Sequence creation ───────────────────────────────────────────────────────

def create_sequences(df, seq_length=None):
    """
    Create overlapping sequences of length *seq_length* for LSTM input.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the feature columns **already scaled** and a
        ``state_label`` column with string labels.
    seq_length : int, optional
        Defaults to ``config.SEQUENCE_LENGTH``.

    Returns
    -------
    X : np.ndarray of shape ``(n_samples, seq_length, n_features)``
    y : np.ndarray of shape ``(n_samples,)`` with integer labels.
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH

    features = df[FEATURE_COLUMNS].values
    labels = df["state_label"].map(config.STATE_LABELS).values

    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(labels[i + seq_length])

    return np.array(X), np.array(y)


# ── Simulation data ─────────────────────────────────────────────────────────

def generate_simulation_data(n_idle=200, n_low=200, n_high=200, seed=42):
    """
    Generate synthetic simulation data for Idle, Low Load and High Load
    scenarios.  Returns an unlabelled :class:`~pandas.DataFrame`.
    """
    rng = np.random.RandomState(seed)

    def _block(n, cpu, mem, dr, dw, nr, ns, idle):
        return pd.DataFrame({
            "cpu_utilization":    rng.uniform(*cpu, size=n),
            "memory_utilization": rng.uniform(*mem, size=n),
            "disk_read":          rng.uniform(*dr, size=n),
            "disk_write":         rng.uniform(*dw, size=n),
            "network_received":   rng.uniform(*nr, size=n),
            "network_sent":       rng.uniform(*ns, size=n),
            "system_idle_time":   rng.uniform(*idle, size=n),
        })

    idle_df = _block(
        n_idle,
        cpu=(0, 5), mem=(5, 20), dr=(0, 1_000), dw=(0, 1_000),
        nr=(0, 5_000), ns=(0, 5_000), idle=(95, 100),
    )
    low_df = _block(
        n_low,
        cpu=(5, 50), mem=(20, 60), dr=(1_000, 50_000), dw=(1_000, 50_000),
        nr=(5_000, 100_000), ns=(5_000, 100_000), idle=(50, 95),
    )
    high_df = _block(
        n_high,
        cpu=(50, 100), mem=(60, 100), dr=(50_000, 200_000),
        dw=(50_000, 200_000), nr=(100_000, 500_000),
        ns=(100_000, 500_000), idle=(0, 50),
    )

    df = pd.concat([idle_df, low_df, high_df], ignore_index=True)
    return df
