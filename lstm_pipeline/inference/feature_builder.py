import pandas as pd

from lstm_pipeline.features.device_profiler import load_device_profile
from lstm_pipeline.features.feature_engineer import add_time_features


RENAME_MAP = {
    "cpuutilization": "cpu_utilization",
    "memoryutilization": "memory_utilization",
    "diskwrite": "disk_write",
    "diskread": "disk_read",
    "networksent": "network_sent",
    "networkreceived": "network_received",
    "idletime": "idle_time",
}


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_features(df)


def load_device_context(device_name: str, config: dict) -> dict:
    return load_device_profile(device_name, config["paths"]["profiles"])


def add_device_context_to_df(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    out = df.copy()
    for key, value in context.items():
        out[key] = value
    return out


def build_inference_features(df: pd.DataFrame, device_name: str, config: dict) -> pd.DataFrame:
    out = df.rename(columns=RENAME_MAP).copy()
    out["datetime"] = pd.to_datetime(out["date"].astype(str) + " " + out["record_time"].astype(str))
    out = build_time_features(out)
    out = add_device_context_to_df(out, load_device_context(device_name, config))
    return out[config["feature_columns"]]
