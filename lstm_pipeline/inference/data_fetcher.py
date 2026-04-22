from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from lstm_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


RAW_COLUMNS = [
    "record_time",
    "cpuutilization",
    "memoryutilization",
    "diskwrite",
    "diskread",
    "networksent",
    "networkreceived",
    "idletime",
]


def _read_latest_rows(engine, table_name: str, limit: int) -> pd.DataFrame:
    query = text(
        f'SELECT {", ".join(RAW_COLUMNS)} FROM "{table_name}" ORDER BY record_time DESC LIMIT :limit'
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
    return df.sort_values("record_time").reset_index(drop=True)


def fetch_latest_telemetry(engine, device_name: str, lookback: int = 30) -> pd.DataFrame:
    today = date.today()
    yesterday = today - timedelta(days=1)
    tables = [f"{device_name}_{today.strftime('%Y%m%d')}", f"{device_name}_{yesterday.strftime('%Y%m%d')}"]

    frames = []
    remaining = lookback
    for table in tables:
        try:
            df = _read_latest_rows(engine, table, remaining)
            if not df.empty:
                suffix = table.split("_")[-1]
                df["date"] = pd.to_datetime(suffix, format="%Y%m%d").date()
                frames.append(df)
                remaining = lookback - sum(len(x) for x in frames)
                if remaining <= 0:
                    break
        except Exception as exc:
            logger.warning("Could not read table %s: %s", table, exc)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.tail(lookback).reset_index(drop=True)
    return merged
