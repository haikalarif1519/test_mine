"""Main monitoring script that detects idle systems and puts them to sleep.

This script continuously monitors systems via Zabbix, uses the trained AI
model to detect idle systems, and sends sleep commands after the configured
idle timeout (default: 10 minutes).
"""

import argparse
import logging
import signal
import sys
import time

import config
from idle_detection_model import IdleDetectionModel
from zabbix_client import ZabbixClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SleepManager:
    """Monitors systems and puts them to sleep when idle."""

    def __init__(self, model_path=None, dry_run=False):
        self.model = IdleDetectionModel(model_path=model_path)
        self.zabbix = ZabbixClient()
        self.dry_run = dry_run
        self.running = False
        # Track consecutive idle detections per host: host_id -> count
        self.idle_counts = {}

    @property
    def idle_checks_required(self):
        """Number of consecutive idle checks before triggering sleep."""
        return max(1, config.IDLE_THRESHOLD_MINUTES * 60 // config.MONITORING_INTERVAL_SECONDS)

    def start(self, group_name=None):
        """Start the monitoring loop.

        Args:
            group_name: Optional Zabbix host group to monitor.
        """
        self.model.load()
        self.zabbix.login()
        self.running = True

        logger.info(
            "Starting idle monitoring (interval=%ds, idle_threshold=%d checks, dry_run=%s)",
            config.MONITORING_INTERVAL_SECONDS,
            self.idle_checks_required,
            self.dry_run,
        )

        try:
            while self.running:
                self._check_hosts(group_name)
                time.sleep(config.MONITORING_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the monitoring loop and clean up."""
        self.running = False
        try:
            self.zabbix.logout()
        except Exception:
            pass
        logger.info("Monitoring stopped")

    def _check_hosts(self, group_name=None):
        """Check all hosts and manage their idle state."""
        try:
            hosts = self.zabbix.get_hosts(group_name)
        except Exception as e:
            logger.error("Failed to fetch hosts: %s", e)
            return

        for host in hosts:
            host_id = host["hostid"]
            host_name = host["name"]

            try:
                metrics = self.zabbix.get_host_metrics(host_id)
                is_idle, idle_prob = self.model.is_idle(metrics)

                if is_idle:
                    self.idle_counts[host_id] = self.idle_counts.get(host_id, 0) + 1
                    logger.info(
                        "Host %s is IDLE (prob=%.2f, count=%d/%d)",
                        host_name, idle_prob,
                        self.idle_counts[host_id], self.idle_checks_required,
                    )

                    if self.idle_counts[host_id] >= self.idle_checks_required:
                        self._put_to_sleep(host_id, host_name)
                        self.idle_counts[host_id] = 0
                else:
                    if host_id in self.idle_counts and self.idle_counts[host_id] > 0:
                        logger.info(
                            "Host %s is ACTIVE (prob=%.2f), resetting idle counter",
                            host_name, idle_prob,
                        )
                    self.idle_counts[host_id] = 0

            except Exception as e:
                logger.error("Error checking host %s: %s", host_name, e)

    def _put_to_sleep(self, host_id, host_name):
        """Send the sleep command to a host.

        Args:
            host_id: Zabbix host ID.
            host_name: Human-readable host name for logging.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would put host %s to sleep", host_name)
            return

        logger.warning("Putting host %s to sleep", host_name)
        try:
            self.zabbix.execute_remote_command(host_id, config.SLEEP_COMMAND)
            logger.info("Sleep command sent to host %s", host_name)
        except Exception as e:
            logger.error("Failed to put host %s to sleep: %s", host_name, e)


def main():
    parser = argparse.ArgumentParser(
        description="AI-based idle detection and sleep management system",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help="Zabbix host group to monitor",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without actually sending sleep commands",
    )
    args = parser.parse_args()

    manager = SleepManager(model_path=args.model, dry_run=args.dry_run)

    def signal_handler(signum, frame):
        logger.info("Received signal %d, shutting down", signum)
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    manager.start(group_name=args.group)


if __name__ == "__main__":
    main()
