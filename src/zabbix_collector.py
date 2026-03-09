"""
zabbix_collector.py
-------------------
Retrieves system metrics from Zabbix using the Zabbix API (pyzabbix).

Metrics collected per host
--------------------------
- cpu_utilization      : CPU utilisation percentage
- memory_utilization   : Memory utilisation percentage
- disk_read_bps        : Disk read throughput  (bytes / second)
- disk_write_bps       : Disk write throughput (bytes / second)
- network_recv_bps     : Network received throughput (bytes / second)
- network_sent_bps     : Network sent     throughput (bytes / second)
- system_idle_time     : System idle time percentage (from Zabbix agent)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pyzabbix import ZabbixAPI

logger = logging.getLogger(__name__)

# Zabbix item key names as reported by the Zabbix agent
_ITEM_KEYS: Dict[str, str] = {
    "cpu_utilization": "system.cpu.util[,user]",
    "memory_utilization": "vm.memory.size[pused]",
    "disk_read_bps": "vfs.dev.read[,bps]",
    "disk_write_bps": "vfs.dev.write[,bps]",
    "network_recv_bps": "net.if.in[,bps]",
    "network_sent_bps": "net.if.out[,bps]",
    "system_idle_time": "system.cpu.util[,idle]",
}


class ZabbixCollector:
    """Connects to the Zabbix API and retrieves the latest metrics for a list
    of hosts."""

    def __init__(self, url: str, user: str, password: str) -> None:
        self._url = url
        self._user = user
        self._password = password
        self._zapi: Optional[ZabbixAPI] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Authenticate against the Zabbix API."""
        self._zapi = ZabbixAPI(self._url)
        self._zapi.login(self._user, self._password)
        logger.info("Connected to Zabbix API at %s", self._url)

    def disconnect(self) -> None:
        """Log out from the Zabbix API."""
        if self._zapi is not None:
            try:
                self._zapi.user.logout()
            except Exception:  # noqa: BLE001
                pass
            self._zapi = None
            logger.info("Disconnected from Zabbix API")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_host_id(self, hostname: str) -> Optional[str]:
        """Return the Zabbix internal host ID for *hostname*."""
        hosts = self._zapi.host.get(
            filter={"host": hostname},
            output=["hostid"],
        )
        if not hosts:
            logger.warning("Host '%s' not found in Zabbix", hostname)
            return None
        return hosts[0]["hostid"]

    def _get_latest_value(self, host_id: str, item_key: str) -> Optional[float]:
        """Return the most recent numeric value for *item_key* on *host_id*."""
        items = self._zapi.item.get(
            hostids=host_id,
            search={"key_": item_key},
            output=["lastvalue", "lastclock"],
            limit=1,
        )
        if not items:
            logger.debug("Item '%s' not found on host %s", item_key, host_id)
            return None
        try:
            return float(items[0]["lastvalue"])
        except (ValueError, KeyError):
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_metrics(self, hostname: str) -> Dict[str, Any]:
        """Return the latest metrics for *hostname*.

        Returns a dict with keys from ``_ITEM_KEYS`` plus a ``timestamp``
        entry.  Missing values are represented as ``None``.
        """
        if self._zapi is None:
            raise RuntimeError("Not connected to Zabbix API – call connect() first.")

        host_id = self._get_host_id(hostname)
        if host_id is None:
            return {}

        metrics: Dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}
        for metric_name, item_key in _ITEM_KEYS.items():
            metrics[metric_name] = self._get_latest_value(host_id, item_key)

        logger.debug("Collected metrics for '%s': %s", hostname, metrics)
        return metrics

    def get_metrics_for_all(self, hostnames: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return metrics for every hostname in *hostnames*.

        Returns a dict keyed by hostname.
        """
        return {hostname: self.get_metrics(hostname) for hostname in hostnames}
