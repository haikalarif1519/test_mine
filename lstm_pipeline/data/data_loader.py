import re
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from lstm_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    def __init__(self, engine, config: dict):
        self.engine = engine
        self.config = config

    def _table_dates_for_device(self, device: str) -> list[str]:
        pattern = f"{device.lower()}_%"
        query = text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND lower(table_name) LIKE :pattern
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"pattern": pattern}).fetchall()

        start = datetime.strptime(self.config["data"]["start_date"], "%Y-%m-%d").strftime("%Y%m%d")
        end = datetime.strptime(self.config["data"]["end_date"], "%Y-%m-%d").strftime("%Y%m%d")

        valid = []
        for (table_name,) in rows:
            match = re.match(rf"^{re.escape(device)}_(\d{{8}})$", table_name, re.IGNORECASE)
            if not match:
                continue
            day = match.group(1)
            if start <= day <= end:
                valid.append((table_name, day))
        valid.sort(key=lambda x: x[1])
        return valid

    def load_device_data(self, device: str) -> pd.DataFrame:
        table_info = self._table_dates_for_device(device)
        if not table_info:
            logger.warning("No tables found for %s in configured date range", device)
            return pd.DataFrame()

        frames = []
        for table_name, day in table_info:
            try:
                df = pd.read_sql(text(f'SELECT * FROM "{table_name}"'), self.engine)
                df["date"] = pd.to_datetime(day, format="%Y%m%d").date()
                df["device"] = device
                frames.append(df)
            except Exception as exc:
                logger.warning("Skipping missing/unreadable table %s: %s", table_name, exc)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        if "record_time" in merged.columns:
            merged = merged.sort_values(["date", "record_time"]).reset_index(drop=True)
        return merged
