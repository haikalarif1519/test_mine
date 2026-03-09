"""Data preprocessing and feature engineering for idle detection."""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def create_feature_dataframe(metrics_list):
    """Convert a list of metric dictionaries into a pandas DataFrame.

    Args:
        metrics_list: List of dicts, each containing metric values for one
            time point. Keys should match config.FEATURE_COLUMNS.

    Returns:
        pandas DataFrame with feature columns.
    """
    df = pd.DataFrame(metrics_list)
    for col in config.FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[config.FEATURE_COLUMNS]


def label_idle_samples(df, cpu_threshold=10.0, disk_threshold=1.0,
                       network_threshold=1024.0):
    """Label each sample as idle (1) or active (0) based on thresholds.

    This provides rule-based labels that are used to train the ML model.
    A system is considered idle when:
      - CPU utilization is below cpu_threshold %
      - Disk read + write rates are below disk_threshold bytes/s
      - Network received + sent rates are below network_threshold bytes/s

    Args:
        df: DataFrame with feature columns.
        cpu_threshold: CPU usage percent below which the system is idle.
        disk_threshold: Combined disk I/O rate below which the system is idle.
        network_threshold: Combined network rate below which the system is idle.

    Returns:
        numpy array of labels (1 = idle, 0 = active).
    """
    cpu_idle = df["cpu_utilization"] < cpu_threshold
    disk_idle = (df["disk_read"] + df["disk_write"]) < disk_threshold
    network_idle = (df["network_received"] + df["network_sent"]) < network_threshold

    labels = (cpu_idle & disk_idle & network_idle).astype(int).values
    logger.info(
        "Labeled %d samples: %d idle, %d active",
        len(labels), np.sum(labels), len(labels) - np.sum(labels),
    )
    return labels


def normalize_features(df):
    """Normalize feature values to the 0-1 range using min-max scaling.

    Args:
        df: DataFrame with feature columns.

    Returns:
        Tuple of (normalized DataFrame, dict of {column: (min, max)} for
        inverse transform).
    """
    scaling_params = {}
    normalized = df.copy()
    for col in config.FEATURE_COLUMNS:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            normalized[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.0
        scaling_params[col] = (col_min, col_max)
    return normalized, scaling_params


def generate_synthetic_training_data(n_samples=1000, idle_ratio=0.5):
    """Generate synthetic training data for model development and testing.

    This is useful when real Zabbix data is not yet available.

    Args:
        n_samples: Total number of samples to generate.
        idle_ratio: Fraction of samples that should be idle.

    Returns:
        Tuple of (DataFrame with features, numpy array of labels).
    """
    rng = np.random.RandomState(42)
    n_idle = int(n_samples * idle_ratio)
    n_active = n_samples - n_idle

    # Idle systems: low CPU, low disk, low network, high idle time
    idle_data = {
        "cpu_utilization": rng.uniform(0, 8, n_idle),
        "memory_utilization": rng.uniform(10, 40, n_idle),
        "disk_read": rng.uniform(0, 0.5, n_idle),
        "disk_write": rng.uniform(0, 0.5, n_idle),
        "network_received": rng.uniform(0, 500, n_idle),
        "network_sent": rng.uniform(0, 500, n_idle),
        "system_idle_time": rng.uniform(80, 100, n_idle),
    }

    # Active systems: higher CPU, disk, and network activity
    active_data = {
        "cpu_utilization": rng.uniform(15, 100, n_active),
        "memory_utilization": rng.uniform(30, 95, n_active),
        "disk_read": rng.uniform(2, 100, n_active),
        "disk_write": rng.uniform(2, 100, n_active),
        "network_received": rng.uniform(2000, 1_000_000, n_active),
        "network_sent": rng.uniform(2000, 1_000_000, n_active),
        "system_idle_time": rng.uniform(0, 50, n_active),
    }

    idle_df = pd.DataFrame(idle_data)
    active_df = pd.DataFrame(active_data)

    df = pd.concat([idle_df, active_df], ignore_index=True)
    labels = np.array([1] * n_idle + [0] * n_active)

    # Shuffle
    shuffle_idx = rng.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    labels = labels[shuffle_idx]

    return df, labels
