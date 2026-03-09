"""
Main entry-point for the AI-based System Sleep Manager.

Provides sub-commands to:
  • generate and label simulation data
  • train the LSTM model
  • run the sleep-monitoring loop
  • run the Zabbix → PostgreSQL data pipeline
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

import config
from data_preprocessing import (
    FEATURE_COLUMNS,
    create_sequences,
    fit_scaler,
    generate_simulation_data,
    label_dataframe,
    scale_features,
)
from lstm_model import build_model, load_model, save_model, train_model


def cmd_generate(args):
    """Generate simulation data, label it, and save to CSV."""
    logging.info("Generating simulation data …")
    df = generate_simulation_data(
        n_idle=args.n_idle, n_low=args.n_low, n_high=args.n_high,
    )
    df = label_dataframe(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    logging.info("Saved %d rows to %s", len(df), args.output)
    logging.info("Label distribution:\n%s", df["state_label"].value_counts())


def cmd_train(args):
    """Train the LSTM model on labelled data."""
    logging.info("Loading data from %s …", args.data)
    df = pd.read_csv(args.data)

    if "state_label" not in df.columns:
        logging.info("Labelling data …")
        df = label_dataframe(df)

    # Fill NaN values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)

    # Fit scaler and scale
    scaler = fit_scaler(df)
    df_scaled = scale_features(df, scaler)

    # Create sequences
    X, y = create_sequences(df_scaled, seq_length=config.SEQUENCE_LENGTH)
    logging.info("Created %d sequences (shape %s)", len(X), X.shape)

    # Build and train
    model = build_model()
    train_model(
        model, X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save
    save_model(model, config.MODEL_PATH)

    scaler_path = config.SCALER_PATH
    os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logging.info("Scaler saved to %s", scaler_path)


def cmd_monitor(args):
    """Run the real-time sleep monitoring loop."""
    from sleep_manager import SleepManager

    model = load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    manager = SleepManager(model, scaler)
    manager.run()


def cmd_pipeline(args):
    """Run the Zabbix → PostgreSQL data pipeline."""
    from data_pipeline import run_pipeline_loop
    run_pipeline_loop()


def main():
    parser = argparse.ArgumentParser(
        description="AI-based System Sleep Manager",
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    p_gen = sub.add_parser("generate", help="Generate simulation data")
    p_gen.add_argument("--n-idle", type=int, default=200)
    p_gen.add_argument("--n-low", type=int, default=200)
    p_gen.add_argument("--n-high", type=int, default=200)
    p_gen.add_argument("-o", "--output", default="data/simulation_data.csv")
    p_gen.set_defaults(func=cmd_generate)

    # train
    p_train = sub.add_parser("train", help="Train the LSTM model")
    p_train.add_argument("--data", default="data/simulation_data.csv")
    p_train.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p_train.set_defaults(func=cmd_train)

    # monitor
    p_mon = sub.add_parser("monitor", help="Run sleep monitoring loop")
    p_mon.set_defaults(func=cmd_monitor)

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Run Zabbix → DB data pipeline")
    p_pipe.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
