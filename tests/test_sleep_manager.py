"""
Tests for src/sleep_manager.py
"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import paramiko
import pytest

from src.sleep_manager import SleepManager


class TestSleepManagerSendSleep:
    def _make_manager(self, **kwargs):
        defaults = dict(
            username="root",
            password="secret",
            port=22,
            timeout=10,
            known_hosts_file="/tmp/nonexistent_known_hosts",
        )
        defaults.update(kwargs)
        return SleepManager(**defaults)

    def _mock_ssh_success(self):
        """Return a mock SSHClient that simulates a successful sleep command."""
        client = MagicMock(spec=paramiko.SSHClient)
        stdout = MagicMock()
        stdout.channel.recv_exit_status.return_value = 0
        stderr = MagicMock()
        stderr.read.return_value = b""
        client.exec_command.return_value = (MagicMock(), stdout, stderr)
        return client

    def _mock_ssh_failure(self, exit_code=1, stderr_output=b"error"):
        client = MagicMock(spec=paramiko.SSHClient)
        stdout = MagicMock()
        stdout.channel.recv_exit_status.return_value = exit_code
        stderr = MagicMock()
        stderr.read.return_value = stderr_output
        client.exec_command.return_value = (MagicMock(), stdout, stderr)
        return client

    def test_returns_true_on_success(self):
        manager = self._make_manager()
        mock_client = self._mock_ssh_success()

        with patch.object(manager, "_build_client", return_value=mock_client), \
             patch.object(manager, "_connect"):
            result = manager.send_sleep("192.168.1.10")

        assert result is True

    def test_returns_false_on_nonzero_exit(self):
        manager = self._make_manager()
        mock_client = self._mock_ssh_failure(exit_code=1)

        with patch.object(manager, "_build_client", return_value=mock_client), \
             patch.object(manager, "_connect"):
            result = manager.send_sleep("192.168.1.10")

        assert result is False

    def test_returns_false_on_authentication_error(self):
        manager = self._make_manager()
        mock_client = MagicMock(spec=paramiko.SSHClient)

        with patch.object(manager, "_build_client", return_value=mock_client), \
             patch.object(
                 manager, "_connect",
                 side_effect=paramiko.AuthenticationException("auth failed")
             ):
            result = manager.send_sleep("192.168.1.10")

        assert result is False

    def test_returns_false_on_ssh_exception(self):
        manager = self._make_manager()
        mock_client = MagicMock(spec=paramiko.SSHClient)

        with patch.object(manager, "_build_client", return_value=mock_client), \
             patch.object(
                 manager, "_connect",
                 side_effect=paramiko.SSHException("connection error")
             ):
            result = manager.send_sleep("192.168.1.10")

        assert result is False

    def test_returns_false_on_network_error(self):
        manager = self._make_manager()
        mock_client = MagicMock(spec=paramiko.SSHClient)

        with patch.object(manager, "_build_client", return_value=mock_client), \
             patch.object(
                 manager, "_connect",
                 side_effect=OSError("network unreachable")
             ):
            result = manager.send_sleep("192.168.1.10")

        assert result is False

    def test_client_is_always_closed(self):
        manager = self._make_manager()
        mock_client = self._mock_ssh_success()

        with patch.object(manager, "_build_client", return_value=mock_client), \
             patch.object(manager, "_connect"):
            manager.send_sleep("192.168.1.10")

        mock_client.close.assert_called_once()

    def test_client_closed_even_on_error(self):
        manager = self._make_manager()
        mock_client = MagicMock(spec=paramiko.SSHClient)

        with patch.object(manager, "_build_client", return_value=mock_client), \
             patch.object(
                 manager, "_connect",
                 side_effect=OSError("fail")
             ):
            manager.send_sleep("192.168.1.10")

        mock_client.close.assert_called_once()

    def test_send_sleep_safe_returns_none_on_unexpected_error(self):
        manager = self._make_manager()

        with patch.object(
            manager, "_build_client", side_effect=RuntimeError("unexpected")
        ):
            result = manager.send_sleep_safe("192.168.1.10")

        assert result is None

    def test_uses_key_file_when_provided(self):
        manager = self._make_manager(key_file="/home/user/.ssh/id_rsa")
        mock_client = MagicMock(spec=paramiko.SSHClient)
        mock_client.connect = MagicMock()
        stdout = MagicMock()
        stdout.channel.recv_exit_status.return_value = 0
        stderr = MagicMock()
        stderr.read.return_value = b""
        mock_client.exec_command.return_value = (MagicMock(), stdout, stderr)

        with patch.object(manager, "_build_client", return_value=mock_client):
            manager.send_sleep("192.168.1.10")

        call_kwargs = mock_client.connect.call_args.kwargs
        assert "key_filename" in call_kwargs
        assert "password" not in call_kwargs

    def test_uses_reject_policy_when_known_hosts_missing(self):
        manager = self._make_manager(known_hosts_file="/tmp/nonexistent_known_hosts_xyz")
        with patch("paramiko.SSHClient") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            manager._build_client()
            mock_instance.set_missing_host_key_policy.assert_called_once()
            policy_arg = mock_instance.set_missing_host_key_policy.call_args[0][0]
            assert isinstance(policy_arg, paramiko.RejectPolicy)

    def test_loads_known_hosts_when_file_exists(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix="known_hosts", delete=False) as f:
            known_hosts_path = f.name
        try:
            manager = self._make_manager(known_hosts_file=known_hosts_path)
            with patch("paramiko.SSHClient") as MockClient:
                mock_instance = MagicMock()
                MockClient.return_value = mock_instance
                manager._build_client()
                mock_instance.load_host_keys.assert_called_once_with(known_hosts_path)
        finally:
            os.unlink(known_hosts_path)
