import torch
import pandas as pd

from lstm_pipeline.data.data_loader import DataLoader
from lstm_pipeline.data.db_connector import get_engine
from lstm_pipeline.data.preprocessor import Preprocessor
from lstm_pipeline.data.sequence_builder import create_sequences, oversample_minority_sequences, train_val_test_split
from lstm_pipeline.evaluation.evaluator import evaluate_model
from lstm_pipeline.evaluation.visualizer import plot_class_distribution, plot_confusion_matrix, plot_loss_curve
from lstm_pipeline.features.device_profiler import compute_device_context, save_device_profile
from lstm_pipeline.features.feature_engineer import add_time_features
from lstm_pipeline.model.lstm_model import build_lstm_model
from lstm_pipeline.model.trainer import train_model
from lstm_pipeline.utils.helpers import ensure_dirs, load_config
from lstm_pipeline.utils.logger import get_logger

logger = get_logger(__name__)

RENAME_MAP = {
    "cpuutilization": "cpu_utilization",
    "memoryutilization": "memory_utilization",
    "diskwrite": "disk_write",
    "diskread": "disk_read",
    "networksent": "network_sent",
    "networkreceived": "network_received",
    "idletime": "idle_time",
}


def main():
    config = load_config()
    ensure_dirs(config)

    engine = get_engine(config)
    loader = DataLoader(engine, config)
    preprocessor = Preprocessor(config)

    device_frames = []
    for device in config["devices"]:
        logger.info("Loading data for %s", device)
        df = loader.load_device_data(device)
        if df.empty:
            logger.warning("No data for %s; skipping", device)
            continue

        df = df.rename(columns=RENAME_MAP)
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["record_time"].astype(str))
        df = add_time_features(df)

        profile = compute_device_context(df)
        save_device_profile(profile, device, config["paths"]["profiles"])
        for k, v in profile.items():
            df[k] = v

        device_frames.append(df)

    if not device_frames:
        logger.error("No data loaded for any configured device")
        return

    all_df = pd.concat(device_frames, ignore_index=True).sort_values(["device", "datetime"]).reset_index(drop=True)
    processed, _ = preprocessor.preprocess(all_df, fit_scaler=True)

    plot_class_distribution(processed["label"], config["paths"]["logs"])

    device_sequences = create_sequences(
        processed,
        feature_columns=config["feature_columns"],
        lookback=config["preprocessing"]["lookback"],
        predict_steps=config["preprocessing"]["predict_steps"],
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        device_sequences,
        train_ratio=config["preprocessing"]["train_ratio"],
        val_ratio=config["preprocessing"]["val_ratio"],
        test_ratio=config["preprocessing"]["test_ratio"],
    )

    model = build_lstm_model(config)
    X_train, y_train = oversample_minority_sequences(X_train, y_train)
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    plot_loss_curve(history, config["paths"]["logs"])

    results = evaluate_model(model, X_test, y_test)
    for step, data in results.items():
        plot_confusion_matrix(data["confusion_matrix"], ["Idle", "Low Load", "High Load"], step, config["paths"]["logs"])

    torch.save(model, config["paths"]["model"])
    logger.info("Training complete. Model saved to %s", config["paths"]["model"])


if __name__ == "__main__":
    main()
