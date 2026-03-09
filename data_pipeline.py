"""
Data Pipeline: Zabbix → PostgreSQL.

Pulls system metrics (CPU, memory, disk, network, idle time) from Zabbix
using the Zabbix API and stores them in a PostgreSQL database for
downstream processing by the LSTM model.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2 import sql
from pyzabbix import ZabbixAPI

import config

logger = logging.getLogger(__name__)


# ── Database helpers ────────────────────────────────────────────────────────

def get_db_connection():
    """Return a new PostgreSQL connection."""
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


def init_database():
    """Create the metrics table if it does not already exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS system_metrics (
        id              SERIAL PRIMARY KEY,
        host_id         VARCHAR(64)   NOT NULL,
        timestamp       TIMESTAMP     NOT NULL,
        cpu_utilization FLOAT,
        memory_utilization FLOAT,
        disk_read       FLOAT,
        disk_write      FLOAT,
        network_received FLOAT,
        network_sent    FLOAT,
        system_idle_time FLOAT,
        state_label     VARCHAR(16)
    );
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
        conn.commit()
        logger.info("Database initialised – system_metrics table ready.")
    finally:
        conn.close()


# ── Zabbix helpers ──────────────────────────────────────────────────────────

def get_zabbix_client():
    """Authenticate and return a ZabbixAPI client."""
    zapi = ZabbixAPI(config.ZABBIX_URL)
    zapi.login(config.ZABBIX_USER, config.ZABBIX_PASSWORD)
    logger.info("Connected to Zabbix API (v%s)", zapi.api_version())
    return zapi


def fetch_host_ids(zapi):
    """Return a list of monitored host IDs."""
    hosts = zapi.host.get(output=["hostid", "host"], filter={"status": 0})
    return hosts


def fetch_metrics(zapi, host_id, time_from, time_till):
    """
    Fetch metric history for a single host between *time_from* and
    *time_till* (UNIX timestamps).

    Returns a dict keyed by metric name, each value being a list of
    ``(timestamp, value)`` tuples.
    """
    metrics = {}
    for metric_name, item_key in config.ZABBIX_ITEM_KEYS.items():
        items = zapi.item.get(
            hostids=host_id,
            search={"key_": item_key},
            output=["itemid", "value_type"],
        )
        if not items:
            logger.warning(
                "Item key %s not found on host %s", item_key, host_id
            )
            continue

        item = items[0]
        history = zapi.history.get(
            itemids=item["itemid"],
            time_from=time_from,
            time_till=time_till,
            output="extend",
            sortfield="clock",
            sortorder="ASC",
            history=item["value_type"],
        )
        metrics[metric_name] = [
            (int(h["clock"]), float(h["value"])) for h in history
        ]
    return metrics


# ── Storage ─────────────────────────────────────────────────────────────────

def store_metrics(host_id, timestamp, metric_row):
    """Insert a single metric row into PostgreSQL."""
    insert_sql = """
    INSERT INTO system_metrics
        (host_id, timestamp, cpu_utilization, memory_utilization,
         disk_read, disk_write, network_received, network_sent,
         system_idle_time)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(insert_sql, (
                host_id,
                datetime.fromtimestamp(timestamp, tz=timezone.utc),
                metric_row.get("cpu_utilization"),
                metric_row.get("memory_utilization"),
                metric_row.get("disk_read"),
                metric_row.get("disk_write"),
                metric_row.get("network_received"),
                metric_row.get("network_sent"),
                metric_row.get("system_idle_time"),
            ))
        conn.commit()
    finally:
        conn.close()


# ── Pipeline orchestration ──────────────────────────────────────────────────

def _align_metrics(metrics):
    """
    Given a dict of ``{metric_name: [(ts, val), ...]}``, align all metrics
    to a common set of timestamps and yield ``(timestamp, row_dict)``
    tuples.
    """
    # Collect every timestamp that appears in any metric
    all_ts = set()
    for pairs in metrics.values():
        all_ts.update(ts for ts, _ in pairs)

    # Build lookup dicts
    lookups = {
        name: dict(pairs) for name, pairs in metrics.items()
    }

    for ts in sorted(all_ts):
        row = {}
        for name in config.ZABBIX_ITEM_KEYS:
            row[name] = lookups.get(name, {}).get(ts)
        # Only yield if we have at least one value
        if any(v is not None for v in row.values()):
            yield ts, row


def pull_and_store(zapi=None):
    """
    Pull the latest metrics from Zabbix for every host and store them in
    PostgreSQL.
    """
    if zapi is None:
        zapi = get_zabbix_client()

    hosts = fetch_host_ids(zapi)
    time_till = int(time.time())
    time_from = int(
        (datetime.now(tz=timezone.utc)
         - timedelta(hours=config.HISTORY_HOURS)).timestamp()
    )

    for host in hosts:
        host_id = host["hostid"]
        logger.info("Fetching metrics for host %s (%s)", host_id, host["host"])
        metrics = fetch_metrics(zapi, host_id, time_from, time_till)

        count = 0
        for ts, row in _align_metrics(metrics):
            store_metrics(host_id, ts, row)
            count += 1
        logger.info("Stored %d rows for host %s", count, host_id)


def run_pipeline_loop():
    """Run the data pipeline in an infinite loop."""
    init_database()
    zapi = get_zabbix_client()
    logger.info("Starting data pipeline loop (interval=%ds)",
                config.PIPELINE_POLL_INTERVAL_SECONDS)
    while True:
        try:
            pull_and_store(zapi)
        except Exception:
            logger.exception("Pipeline iteration failed")
        time.sleep(config.PIPELINE_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline_loop()
