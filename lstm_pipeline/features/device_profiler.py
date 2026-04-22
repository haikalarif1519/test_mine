import json
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_PROFILE = {
    "avg_active_hours_per_day": 2.0,
    "usage_rate_weekday": 0.4,
    "usage_rate_weekend": 0.1,
    "peak_usage_hour_morning": 1,
    "peak_usage_hour_afternoon": 0,
    "peak_usage_hour_evening": 0,
    "peak_usage_hour_night": 0,
    "usage_consistency_score": 0.5,
}


def compute_device_context(df: pd.DataFrame) -> dict:
    if df.empty:
        return DEFAULT_PROFILE.copy()

    temp = df.copy()
    if "datetime" not in temp.columns:
        temp["datetime"] = pd.to_datetime(temp["date"].astype(str) + " " + temp["record_time"].astype(str))

    active = temp["systemstate"].isin(["Low Load", "High Load"])
    temp["active"] = active.astype(int)
    temp["day"] = temp["datetime"].dt.date
    temp["hour"] = temp["datetime"].dt.hour
    temp["day_of_week"] = temp["datetime"].dt.dayofweek

    daily_active_minutes = temp.groupby("day")["active"].sum()
    avg_active_hours = float(daily_active_minutes.mean() / 60.0) if len(daily_active_minutes) else DEFAULT_PROFILE["avg_active_hours_per_day"]

    weekday = temp[temp["day_of_week"] < 5]
    weekend = temp[temp["day_of_week"] >= 5]
    usage_rate_weekday = float(weekday["active"].mean()) if not weekday.empty else DEFAULT_PROFILE["usage_rate_weekday"]
    usage_rate_weekend = float(weekend["active"].mean()) if not weekend.empty else DEFAULT_PROFILE["usage_rate_weekend"]

    active_by_hour = temp.groupby("hour")["active"].mean().reindex(range(24), fill_value=0.0)
    buckets = {
        "morning": active_by_hour.loc[6:11].mean(),
        "afternoon": active_by_hour.loc[12:16].mean(),
        "evening": active_by_hour.loc[17:21].mean(),
        "night": pd.concat([active_by_hour.loc[22:23], active_by_hour.loc[0:5]]).mean(),
    }
    peak_bucket = max(buckets, key=buckets.get)

    cv = float(daily_active_minutes.std() / daily_active_minutes.mean()) if len(daily_active_minutes) > 1 and daily_active_minutes.mean() > 0 else 0.0
    consistency = float(np.clip(1.0 - cv, 0.0, 1.0))

    return {
        "avg_active_hours_per_day": round(avg_active_hours, 4),
        "usage_rate_weekday": round(usage_rate_weekday, 4),
        "usage_rate_weekend": round(usage_rate_weekend, 4),
        "peak_usage_hour_morning": int(peak_bucket == "morning"),
        "peak_usage_hour_afternoon": int(peak_bucket == "afternoon"),
        "peak_usage_hour_evening": int(peak_bucket == "evening"),
        "peak_usage_hour_night": int(peak_bucket == "night"),
        "usage_consistency_score": round(consistency, 4),
    }


def save_device_profile(profile: dict, device: str, profiles_dir: str) -> Path:
    path = Path(profiles_dir) / f"{device}_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Saved profile for %s to %s", device, path)
    return path


def load_device_profile(device: str, profiles_dir: str) -> dict:
    path = Path(profiles_dir) / f"{device}_profile.json"
    if not path.exists():
        logger.warning("Profile not found for %s, using default profile", device)
        return DEFAULT_PROFILE.copy()
    return json.loads(path.read_text(encoding="utf-8"))
