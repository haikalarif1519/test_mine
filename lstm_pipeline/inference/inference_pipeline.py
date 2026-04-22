from lstm_pipeline.data.db_connector import get_engine
from lstm_pipeline.inference.data_fetcher import fetch_latest_telemetry
from lstm_pipeline.inference.feature_builder import build_inference_features
from lstm_pipeline.inference.normalizer import load_scaler, normalize_inference_features, reshape_for_lstm
from lstm_pipeline.inference.predictor import load_model, predict_trajectory
from lstm_pipeline.utils.helpers import load_config
from lstm_pipeline.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG = load_config()
ENGINE = get_engine(CONFIG)
MODEL = None
SCALER = None

try:
    MODEL = load_model(CONFIG["paths"]["model"])
    SCALER = load_scaler(CONFIG)
except Exception as exc:
    logger.warning("Model/scaler not loaded at startup: %s", exc)


def run_inference(device_name: str):
    try:
        if MODEL is None or SCALER is None:
            logger.error("Model or scaler unavailable. Train model first.")
            return None

        raw = fetch_latest_telemetry(ENGINE, device_name, lookback=CONFIG["preprocessing"]["lookback"])
        if len(raw) < CONFIG["preprocessing"]["lookback"]:
            logger.warning("Insufficient rows for %s: expected %s, got %s", device_name, CONFIG["preprocessing"]["lookback"], len(raw))
            return None

        feats = build_inference_features(raw, device_name, CONFIG)
        norm = normalize_inference_features(feats, SCALER, CONFIG)
        x_input = reshape_for_lstm(norm)
        return predict_trajectory(MODEL, x_input)
    except Exception as exc:
        logger.exception("Inference failed for %s: %s", device_name, exc)
        return None
