"""Zabbix API client for retrieving system metrics."""

import logging
import requests

import config

logger = logging.getLogger(__name__)


class ZabbixClient:
    """Client for interacting with the Zabbix API to collect system metrics."""

    def __init__(self, url=None, user=None, password=None):
        self.url = url or config.ZABBIX_URL
        self.user = user or config.ZABBIX_USER
        self.password = password or config.ZABBIX_PASSWORD
        self.auth_token = None
        self._request_id = 0

    def _next_request_id(self):
        self._request_id += 1
        return self._request_id

    def _api_request(self, method, params):
        """Send a JSON-RPC request to the Zabbix API."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._next_request_id(),
        }
        if self.auth_token:
            payload["auth"] = self.auth_token

        response = requests.post(
            self.url,
            json=payload,
            headers={"Content-Type": "application/json-rpc"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            raise RuntimeError(
                f"Zabbix API error: {result['error']['data']}"
            )
        return result.get("result")

    def login(self):
        """Authenticate with the Zabbix API and store the auth token."""
        self.auth_token = self._api_request(
            "user.login",
            {"user": self.user, "password": self.password},
        )
        logger.info("Successfully authenticated with Zabbix API")
        return self.auth_token

    def get_hosts(self, group_name=None):
        """Retrieve monitored hosts, optionally filtered by host group."""
        params = {"output": ["hostid", "host", "name"], "selectInterfaces": ["ip"]}
        if group_name:
            groups = self._api_request(
                "hostgroup.get",
                {"output": ["groupid"], "filter": {"name": [group_name]}},
            )
            if groups:
                params["groupids"] = [g["groupid"] for g in groups]
        return self._api_request("host.get", params)

    def get_item_latest_value(self, host_id, item_key):
        """Get the latest value for a specific item key on a host."""
        items = self._api_request(
            "item.get",
            {
                "output": ["itemid", "lastvalue", "lastclock"],
                "hostids": host_id,
                "search": {"key_": item_key},
                "limit": 1,
            },
        )
        if items:
            return float(items[0].get("lastvalue", 0))
        return None

    def get_item_history(self, host_id, item_key, time_from, time_till, limit=1000):
        """Get historical values for a specific item key on a host."""
        items = self._api_request(
            "item.get",
            {
                "output": ["itemid"],
                "hostids": host_id,
                "search": {"key_": item_key},
                "limit": 1,
            },
        )
        if not items:
            return []

        item_id = items[0]["itemid"]
        history = self._api_request(
            "history.get",
            {
                "output": "extend",
                "itemids": item_id,
                "history": 0,  # 0 = float
                "time_from": time_from,
                "time_till": time_till,
                "limit": limit,
                "sortfield": "clock",
                "sortorder": "ASC",
            },
        )
        return [{"timestamp": int(h["clock"]), "value": float(h["value"])} for h in history]

    def get_host_metrics(self, host_id):
        """Get all relevant metrics for a host as a dictionary."""
        metrics = {}
        for metric_name, item_key in config.ZABBIX_ITEM_KEYS.items():
            value = self.get_item_latest_value(host_id, item_key)
            metrics[metric_name] = value if value is not None else 0.0
        return metrics

    def execute_remote_command(self, host_id, command):
        """Execute a command on a remote host via Zabbix agent."""
        result = self._api_request(
            "script.execute",
            {
                "scriptid": "0",
                "hostid": host_id,
                "command": command,
            },
        )
        logger.info("Executed command on host %s: %s", host_id, command)
        return result

    def logout(self):
        """End the Zabbix API session."""
        if self.auth_token:
            self._api_request("user.logout", [])
            self.auth_token = None
            logger.info("Logged out from Zabbix API")
