"""
Tests for src/zabbix_collector.py
"""
from unittest.mock import MagicMock, patch

import pytest

from src.zabbix_collector import ZabbixCollector, _ITEM_KEYS


class TestZabbixCollectorConnection:
    def test_connect_calls_login(self):
        with patch("src.zabbix_collector.ZabbixAPI") as MockAPI:
            mock_instance = MagicMock()
            MockAPI.return_value = mock_instance

            collector = ZabbixCollector("http://zabbix", "Admin", "secret")
            collector.connect()

            MockAPI.assert_called_once_with("http://zabbix")
            mock_instance.login.assert_called_once_with("Admin", "secret")

    def test_disconnect_calls_logout(self):
        with patch("src.zabbix_collector.ZabbixAPI") as MockAPI:
            mock_instance = MagicMock()
            MockAPI.return_value = mock_instance

            collector = ZabbixCollector("http://zabbix", "Admin", "secret")
            collector.connect()
            collector.disconnect()

            mock_instance.user.logout.assert_called_once()

    def test_disconnect_without_connect_is_safe(self):
        collector = ZabbixCollector("http://zabbix", "Admin", "secret")
        collector.disconnect()  # should not raise

    def test_get_metrics_raises_when_not_connected(self):
        collector = ZabbixCollector("http://zabbix", "Admin", "secret")
        with pytest.raises(RuntimeError, match="Not connected"):
            collector.get_metrics("SUT-01")


class TestZabbixCollectorGetMetrics:
    def _make_collector_with_mock_api(self):
        with patch("src.zabbix_collector.ZabbixAPI") as MockAPI:
            mock_api = MagicMock()
            MockAPI.return_value = mock_api

            collector = ZabbixCollector("http://zabbix", "Admin", "secret")
            collector.connect()
            return collector, mock_api

    def test_returns_empty_dict_when_host_not_found(self):
        collector, mock_api = self._make_collector_with_mock_api()
        mock_api.host.get.return_value = []

        result = collector.get_metrics("UNKNOWN-HOST")

        assert result == {}

    def test_returns_metrics_with_expected_keys(self):
        collector, mock_api = self._make_collector_with_mock_api()
        mock_api.host.get.return_value = [{"hostid": "101"}]
        mock_api.item.get.return_value = [{"lastvalue": "42.5", "lastclock": "1000"}]

        result = collector.get_metrics("SUT-01")

        assert "timestamp" in result
        for key in _ITEM_KEYS:
            assert key in result

    def test_missing_item_returns_none_value(self):
        collector, mock_api = self._make_collector_with_mock_api()
        mock_api.host.get.return_value = [{"hostid": "101"}]
        # Simulate item not found
        mock_api.item.get.return_value = []

        result = collector.get_metrics("SUT-01")

        for key in _ITEM_KEYS:
            assert result[key] is None

    def test_get_metrics_for_all_calls_get_metrics_per_host(self):
        collector, mock_api = self._make_collector_with_mock_api()
        mock_api.host.get.return_value = [{"hostid": "101"}]
        mock_api.item.get.return_value = [{"lastvalue": "10.0", "lastclock": "1000"}]

        result = collector.get_metrics_for_all(["SUT-01", "NUC-01"])

        assert set(result.keys()) == {"SUT-01", "NUC-01"}
