"""Tests for the SleepManager class."""

from unittest.mock import MagicMock, patch

import pytest

import config
from main import SleepManager


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.is_idle.return_value = (True, 0.95)
    return model


@pytest.fixture
def mock_zabbix():
    zabbix = MagicMock()
    zabbix.get_hosts.return_value = [
        {"hostid": "1001", "name": "SUT-01", "host": "sut01"},
    ]
    zabbix.get_host_metrics.return_value = {
        "cpu_utilization": 2.0,
        "memory_utilization": 20.0,
        "disk_read": 0.1,
        "disk_write": 0.1,
        "network_received": 50.0,
        "network_sent": 30.0,
        "system_idle_time": 98.0,
    }
    return zabbix


class TestSleepManager:
    def test_idle_checks_required(self):
        manager = SleepManager(dry_run=True)
        expected = config.IDLE_THRESHOLD_MINUTES * 60 // config.MONITORING_INTERVAL_SECONDS
        assert manager.idle_checks_required == expected

    def test_increments_idle_count(self, mock_model, mock_zabbix):
        manager = SleepManager(dry_run=True)
        manager.model = mock_model
        manager.zabbix = mock_zabbix

        manager._check_hosts()
        assert manager.idle_counts["1001"] == 1

        manager._check_hosts()
        assert manager.idle_counts["1001"] == 2

    def test_resets_idle_count_when_active(self, mock_model, mock_zabbix):
        manager = SleepManager(dry_run=True)
        manager.model = mock_model
        manager.zabbix = mock_zabbix

        # First check: idle
        manager._check_hosts()
        assert manager.idle_counts["1001"] == 1

        # Second check: active
        mock_model.is_idle.return_value = (False, 0.2)
        manager._check_hosts()
        assert manager.idle_counts["1001"] == 0

    def test_triggers_sleep_after_threshold(self, mock_model, mock_zabbix):
        manager = SleepManager(dry_run=True)
        manager.model = mock_model
        manager.zabbix = mock_zabbix

        # Simulate enough idle checks to trigger sleep
        for _ in range(manager.idle_checks_required):
            manager._check_hosts()

        # After triggering sleep, counter is reset
        assert manager.idle_counts["1001"] == 0

    def test_dry_run_does_not_execute_command(self, mock_model, mock_zabbix):
        manager = SleepManager(dry_run=True)
        manager.model = mock_model
        manager.zabbix = mock_zabbix

        # Set counter just below threshold, then check once more
        manager.idle_counts["1001"] = manager.idle_checks_required - 1
        manager._check_hosts()

        # Sleep command should NOT have been executed (dry run)
        mock_zabbix.execute_remote_command.assert_not_called()

    def test_non_dry_run_executes_command(self, mock_model, mock_zabbix):
        manager = SleepManager(dry_run=False)
        manager.model = mock_model
        manager.zabbix = mock_zabbix

        # Set counter just below threshold, then check once more
        manager.idle_counts["1001"] = manager.idle_checks_required - 1
        manager._check_hosts()

        # Sleep command should have been executed
        mock_zabbix.execute_remote_command.assert_called_once_with(
            "1001", config.SLEEP_COMMAND,
        )

    def test_handles_zabbix_error_gracefully(self, mock_model, mock_zabbix):
        manager = SleepManager(dry_run=True)
        manager.model = mock_model
        manager.zabbix = mock_zabbix

        mock_zabbix.get_hosts.side_effect = RuntimeError("Connection failed")
        # Should not raise
        manager._check_hosts()
