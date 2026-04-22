from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import yaml
from tensorflow.keras.models import load_model as keras_load_model


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(path: str | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_date_list(start: str, end: str) -> List[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def ensure_dirs(config: dict) -> None:
    for key in ("saved_models", "logs", "profiles"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def load_saved_model(config: dict):
    return keras_load_model(config["paths"]["model"])
