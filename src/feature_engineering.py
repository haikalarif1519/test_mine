"""
feature_engineering.py
-----------------------
Transforms raw Zabbix metrics into a feature vector suitable for the AI model.

The feature vector (in order) is:
  [cpu_utilization, memory_utilization, disk_read_bps, disk_write_bps,
   network_recv_bps, network_sent_bps, system_idle_time]
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: List[str] = [
    "cpu_utilization",
    "memory_utilization",
    "disk_read_bps",
    "disk_write_bps",
    "network_recv_bps",
    "network_sent_bps",
    "system_idle_time",
]


def metrics_to_feature_vector(metrics: Dict[str, Any]) -> Optional[np.ndarray]:
    """Convert a raw metrics dict to a 1-D NumPy feature vector.

    Returns *None* if any required field is missing or ``None``.
    """
    values = []
    for col in FEATURE_COLUMNS:
        val = metrics.get(col)
        if val is None:
            logger.warning("Missing metric '%s' – cannot build feature vector", col)
            return None
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            logger.warning("Non-numeric value for metric '%s': %r", col, val)
            return None
    return np.array(values, dtype=np.float64)


def metrics_to_dataframe(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of raw metrics dicts to a :class:`pandas.DataFrame`.

    Rows with missing values are dropped and a warning is logged.
    """
    rows = []
    for m in metrics_list:
        row = {col: m.get(col) for col in FEATURE_COLUMNS}
        rows.append(row)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d row(s) with missing metric values", dropped)
    return df.astype(np.float64)


def label_idle(
    df: pd.DataFrame,
    cpu_idle_percent: float = 5.0,
    disk_read_bps: float = 102_400.0,
    disk_write_bps: float = 102_400.0,
    network_recv_bps: float = 51_200.0,
    network_sent_bps: float = 51_200.0,
    system_idle_time_percent: float = 95.0,
) -> pd.Series:
    """Apply rule-based labelling to produce a binary ``is_idle`` column.

    A sample is labelled **idle (1)** when *all* of the following hold:
    - cpu_utilization   < *cpu_idle_percent*
    - disk_read_bps     < *disk_read_bps*
    - disk_write_bps    < *disk_write_bps*
    - network_recv_bps  < *network_recv_bps*
    - network_sent_bps  < *network_sent_bps*
    - system_idle_time  > *system_idle_time_percent*
    """
    idle_mask = (
        (df["cpu_utilization"] < cpu_idle_percent)
        & (df["disk_read_bps"] < disk_read_bps)
        & (df["disk_write_bps"] < disk_write_bps)
        & (df["network_recv_bps"] < network_recv_bps)
        & (df["network_sent_bps"] < network_sent_bps)
        & (df["system_idle_time"] > system_idle_time_percent)
    )
    return idle_mask.astype(int).rename("is_idle")
