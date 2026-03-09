"""Script to train the idle detection model.

This script can use either:
  - Historical data fetched from Zabbix
  - Synthetic data for initial model development

Usage:
    python train_model.py --synthetic           # Train with synthetic data
    python train_model.py --zabbix --group LAB  # Train with Zabbix data
"""

import argparse
import logging
import os
import time

import pandas as pd

import config
from data_preprocessing import (
    create_feature_dataframe,
    generate_synthetic_training_data,
    label_idle_samples,
)
from idle_detection_model import IdleDetectionModel
from zabbix_client import ZabbixClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_training_data_from_zabbix(group_name=None):
    """Fetch historical metric data from Zabbix for training.

    Args:
        group_name: Optional host group to filter by.

    Returns:
        Tuple of (DataFrame with features, numpy array of labels).
    """
    client = ZabbixClient()
    client.login()

    try:
        hosts = client.get_hosts(group_name)
        logger.info("Found %d hosts", len(hosts))

        all_metrics = []
        time_till = int(time.time())
        time_from = time_till - config.HISTORY_PERIOD_SECONDS

        for host in hosts:
            host_id = host["hostid"]
            host_name = host["name"]
            logger.info("Fetching data for host: %s", host_name)

            # Fetch history for each metric
            metric_histories = {}
            for metric_name, item_key in config.ZABBIX_ITEM_KEYS.items():
                history = client.get_item_history(
                    host_id, item_key, time_from, time_till,
                    limit=config.HISTORY_LIMIT,
                )
                metric_histories[metric_name] = {
                    h["timestamp"]: h["value"] for h in history
                }

            # Align timestamps across all metrics
            if metric_histories:
                all_timestamps = set()
                for hist in metric_histories.values():
                    all_timestamps.update(hist.keys())

                for ts in sorted(all_timestamps):
                    sample = {}
                    has_all = True
                    for metric_name in config.FEATURE_COLUMNS:
                        value = metric_histories.get(metric_name, {}).get(ts)
                        if value is None:
                            has_all = False
                            break
                        sample[metric_name] = value
                    if has_all:
                        all_metrics.append(sample)

        if not all_metrics:
            logger.warning("No training data collected from Zabbix")
            return None, None

        df = create_feature_dataframe(all_metrics)
        labels = label_idle_samples(df)
        return df, labels
    finally:
        client.logout()


def main():
    parser = argparse.ArgumentParser(description="Train the idle detection model")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for training",
    )
    parser.add_argument(
        "--zabbix", action="store_true",
        help="Fetch training data from Zabbix",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help="Zabbix host group name to filter by",
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the trained model",
    )
    args = parser.parse_args()

    if args.synthetic:
        logger.info("Generating %d synthetic training samples", args.samples)
        X, y = generate_synthetic_training_data(n_samples=args.samples)
    elif args.zabbix:
        logger.info("Fetching training data from Zabbix")
        X, y = fetch_training_data_from_zabbix(group_name=args.group)
        if X is None:
            logger.error("Failed to fetch training data from Zabbix")
            return
    else:
        parser.print_help()
        return

    model = IdleDetectionModel(model_path=args.output)
    results = model.train(X, y)
    logger.info("Training accuracy: %.4f", results["accuracy"])

    model_path = args.output or config.MODEL_PATH
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Show feature importance
    importance = model.get_feature_importance()
    logger.info("Feature importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        logger.info("  %s: %.4f", feature, score)


if __name__ == "__main__":
    main()
