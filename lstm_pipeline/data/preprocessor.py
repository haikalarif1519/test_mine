from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from lstm_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    def __init__(self, config: dict):
        self.config = config

    def build_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["datetime"] = pd.to_datetime(out["date"].astype(str) + " " + out["record_time"].astype(str))
        return out.sort_values(["device", "datetime"]).reset_index(drop=True)

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        gap_minutes = self.config.get("data", {}).get("power_cycle_gap_minutes", 30)
        gap_threshold = pd.Timedelta(minutes=gap_minutes)

        # Hardware metric columns to zero out during a power-cycle gap.
        # These are physically 0 when the device is off; keeping forward-filled
        # Idle values would make Power Cycle rows indistinguishable from Idle.
        hw_cols = [
            c for c in [
                "cpu_utilization", "memory_utilization",
                "disk_write", "disk_read",
                "network_sent", "network_received",
                "idle_time",
            ]
            if c in df.columns
        ]

        per_device = []
        for device, part in df.groupby("device", sort=False):
            part = part.set_index("datetime").sort_index()

            # Detect gaps before resampling so we know which synthetic rows to relabel.
            times = part.index
            diffs = pd.Series(times, index=times).diff()
            gap_end_times = times[diffs > gap_threshold]
            gap_intervals = []
            for gap_end in gap_end_times:
                gap_start = times[times < gap_end][-1]
                gap_intervals.append((gap_start, gap_end))
                logger.info(
                    "Device %s: detected power-cycle gap %s → %s (%d min)",
                    device,
                    gap_start,
                    gap_end,
                    int((gap_end - gap_start).total_seconds() / 60),
                )

            part = part.resample("1min").ffill()
            part["device"] = device
            part["date"] = part.index.date

            # Relabel synthetic rows inside each gap as Power Cycle.
            for gap_start, gap_end in gap_intervals:
                mask = (part.index > gap_start) & (part.index < gap_end)
                part.loc[mask, "systemstate"] = "Power Cycle"
                for col in hw_cols:
                    part.loc[mask, col] = 0.0

            per_device.append(part.reset_index())
        return pd.concat(per_device, ignore_index=True)

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["label"] = out["systemstate"].map(self.config["label_map"]).astype("Int64")
        out = out.dropna(subset=["label"]).copy()
        out["label"] = out["label"].astype(int)
        return out

    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> tuple[pd.DataFrame, MinMaxScaler]:
        out = df.copy()
        scaler_path = Path(self.config["paths"]["scaler"])
        scale_cols = self.config["scale_columns"]

        if fit:
            scaler = MinMaxScaler()
            out[scale_cols] = scaler.fit_transform(out[scale_cols].astype(float))
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info("Saved scaler to %s", scaler_path)
        else:
            scaler = joblib.load(scaler_path)
            out[scale_cols] = scaler.transform(out[scale_cols].astype(float))
        return out, scaler

    def preprocess(self, df: pd.DataFrame, fit_scaler: bool = True):
        out = self.build_datetime(df)
        out = self.handle_missing(out)
        out = self.encode_labels(out)
        out = out.dropna(subset=self.config["feature_columns"]).copy()
        out, scaler = self.normalize_features(out, fit=fit_scaler)
        return out, scaler
