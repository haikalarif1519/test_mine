import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" not in out.columns:
        out["datetime"] = pd.to_datetime(out["date"].astype(str) + " " + out["record_time"].astype(str))

    out["hour"] = out["datetime"].dt.hour.astype(int)
    out["day_of_week"] = out["datetime"].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_working_hour"] = out["hour"].between(8, 18, inclusive="left").astype(int)
    return out
