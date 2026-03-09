"""
main.py
-------
Entry point for the AI-powered lab sleep management system.

Usage
-----
    python main.py [--config config.yaml] [--train] [--log-level DEBUG]

Flags
~~~~~
--config      Path to YAML configuration file (default: config.yaml)
--train       Bootstrap / retrain the model before starting the monitoring
              loop.  Useful on first run.
--log-level   Python logging level (DEBUG / INFO / WARNING / ERROR).
"""

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import yaml

from src.model import IdleDetectionModel
from src.sleep_manager import SleepManager
from src.zabbix_collector import ZabbixCollector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def bootstrap_training_data(
    collector: ZabbixCollector,
    hostnames: List[str],
    n_samples: int = 50,
    poll_interval: int = 5,
) -> List[Dict[str, Any]]:
    """Collect *n_samples* metric snapshots across all hosts for bootstrapping
    the model.  Each snapshot is collected every *poll_interval* seconds."""
    logger.info(
        "Collecting %d bootstrap samples (one every %ds) …",
        n_samples,
        poll_interval,
    )
    all_metrics: List[Dict[str, Any]] = []
    for i in range(n_samples):
        for hostname in hostnames:
            metrics = collector.get_metrics(hostname)
            if metrics:
                all_metrics.append(metrics)
        if i < n_samples - 1:
            time.sleep(poll_interval)
    return all_metrics


def load_csv_training_data(csv_path: str) -> List[Dict[str, Any]]:
    """Load training data from a CSV file.

    Expected columns: cpu_utilization, memory_utilization, disk_read_bps,
    disk_write_bps, network_recv_bps, network_sent_bps, system_idle_time
    (and optionally: is_idle)
    """
    records: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(dict(row))
    logger.info("Loaded %d training rows from '%s'", len(records), csv_path)
    return records


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------


def run_monitor(
    collector: ZabbixCollector,
    model: IdleDetectionModel,
    sleep_manager: SleepManager,
    systems: List[Dict[str, Any]],
    poll_interval: int,
    idle_duration: int,
) -> None:
    """Run the monitoring loop indefinitely.

    For each poll cycle, collect metrics for every system, run the idle
    prediction, and put systems that have been idle for *idle_duration*
    seconds to sleep.
    """
    # Track consecutive idle seconds per system
    idle_since: Dict[str, datetime] = {}

    logger.info(
        "Starting monitoring loop (poll_interval=%ds, idle_duration=%ds) …",
        poll_interval,
        idle_duration,
    )

    while True:
        cycle_start = time.monotonic()

        for system in systems:
            name = system["name"]
            host = system["host"]
            zabbix_hostname = system.get("zabbix_hostname", name)

            try:
                metrics = collector.get_metrics(zabbix_hostname)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to collect metrics for '%s': %s", name, exc)
                continue

            if not metrics:
                logger.warning("No metrics returned for '%s'; skipping.", name)
                continue

            is_idle = model.predict(metrics)

            if is_idle:
                if name not in idle_since:
                    idle_since[name] = datetime.now(timezone.utc)
                    logger.info("System '%s' entered idle state.", name)
                else:
                    elapsed = (datetime.now(timezone.utc) - idle_since[name]).total_seconds()
                    logger.debug(
                        "System '%s' idle for %.0f / %d seconds.",
                        name,
                        elapsed,
                        idle_duration,
                    )
                    if elapsed >= idle_duration:
                        logger.info(
                            "System '%s' has been idle for %.0f seconds – sending sleep.",
                            name,
                            elapsed,
                        )
                        success = sleep_manager.send_sleep(host)
                        if success:
                            # Reset idle timer after successful sleep command
                            del idle_since[name]
            else:
                if name in idle_since:
                    logger.info("System '%s' became active again.", name)
                    del idle_since[name]

        elapsed_cycle = time.monotonic() - cycle_start
        sleep_time = max(0.0, poll_interval - elapsed_cycle)
        time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI-powered lab sleep management system"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="(Re-)train the model before starting the monitoring loop",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    if not os.path.exists(args.config):
        logger.error("Configuration file '%s' not found.", args.config)
        return 1

    cfg = load_config(args.config)

    # -----------------------------------------------------------------------
    # Build components from config
    # -----------------------------------------------------------------------
    zabbix_cfg = cfg["zabbix"]
    collector = ZabbixCollector(
        url=zabbix_cfg["url"],
        user=zabbix_cfg["user"],
        password=zabbix_cfg["password"],
    )

    ssh_cfg = cfg["ssh"]
    sleep_manager = SleepManager(
        username=ssh_cfg["username"],
        password=ssh_cfg.get("password", ""),
        key_file=ssh_cfg.get("key_file", ""),
        port=ssh_cfg.get("port", 22),
        timeout=ssh_cfg.get("timeout", 10),
        known_hosts_file=os.path.expanduser(ssh_cfg.get("known_hosts_file", "~/.ssh/known_hosts")),
    )

    model_cfg = cfg["model"]
    model = IdleDetectionModel(
        model_path=model_cfg["model_path"],
        idle_threshold=model_cfg["idle_threshold"],
    )

    idle_cfg = cfg["idle_detection"]
    poll_interval = idle_cfg["poll_interval_seconds"]
    idle_duration = idle_cfg["idle_duration_seconds"]

    systems = cfg["systems"]
    hostnames = [s.get("zabbix_hostname", s["name"]) for s in systems]

    thresholds = cfg.get("thresholds", {})

    # -----------------------------------------------------------------------
    # Model training
    # -----------------------------------------------------------------------
    if args.train or not os.path.exists(model_cfg["model_path"]):
        logger.info("Training / retraining the idle-detection model …")
        try:
            collector.connect()

            training_data_path = model_cfg.get("training_data_path", "")
            if training_data_path and os.path.exists(training_data_path):
                metrics_list = load_csv_training_data(training_data_path)
            else:
                logger.info(
                    "No pre-labelled training data found – collecting live "
                    "bootstrap samples from Zabbix."
                )
                metrics_list = bootstrap_training_data(
                    collector, hostnames, n_samples=50, poll_interval=5
                )

            train_metrics = model.train_from_metrics_list(
                metrics_list, thresholds=thresholds
            )
            logger.info("Training complete: %s", train_metrics)
            model.save()
        finally:
            collector.disconnect()
    else:
        logger.info("Loading existing model from '%s'.", model_cfg["model_path"])
        model.load()

    # -----------------------------------------------------------------------
    # Monitoring loop
    # -----------------------------------------------------------------------
    try:
        collector.connect()
        run_monitor(
            collector=collector,
            model=model,
            sleep_manager=sleep_manager,
            systems=systems,
            poll_interval=poll_interval,
            idle_duration=idle_duration,
        )
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
    finally:
        collector.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
