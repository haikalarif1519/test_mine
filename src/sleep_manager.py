"""
sleep_manager.py
----------------
Sends a system-sleep (suspend) command to a remote host over SSH.

The ``systemctl suspend`` command is used which works on any modern Linux
system.  For Windows targets the command can be changed to
``rundll32.exe powrprof.dll,SetSuspendState 0,1,0``.
"""

import logging
import os
from typing import Optional

import paramiko

logger = logging.getLogger(__name__)

_SLEEP_COMMAND = "systemctl suspend"
_DEFAULT_KNOWN_HOSTS = os.path.expanduser("~/.ssh/known_hosts")


class SleepManager:
    """Sends sleep commands to remote systems via SSH.

    Parameters
    ----------
    username:
        SSH login username.
    password:
        SSH password.  Ignored when *key_file* is provided.
    key_file:
        Path to an RSA/Ed25519 private key file.  When given, key-based
        authentication is used instead of password authentication.
    port:
        SSH port (default 22).
    timeout:
        TCP connection timeout in seconds (default 10).
    known_hosts_file:
        Path to the SSH known_hosts file used to verify host keys.  Defaults
        to ``~/.ssh/known_hosts``.  The file is loaded when it exists.  If
        the remote host key is not present in the file the connection is
        rejected (``RejectPolicy``), preventing man-in-the-middle attacks.
    """

    def __init__(
        self,
        username: str,
        password: str = "",
        key_file: str = "",
        port: int = 22,
        timeout: int = 10,
        known_hosts_file: str = _DEFAULT_KNOWN_HOSTS,
    ) -> None:
        self.username = username
        self.password = password
        self.key_file = key_file
        self.port = port
        self.timeout = timeout
        self.known_hosts_file = known_hosts_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        # Load known host keys for server verification
        if self.known_hosts_file and os.path.exists(self.known_hosts_file):
            client.load_host_keys(self.known_hosts_file)
        else:
            logger.warning(
                "Known-hosts file '%s' not found – host key verification will "
                "reject all connections.  Add target hosts to your known_hosts "
                "file before running.",
                self.known_hosts_file,
            )
        # Reject connections to hosts whose key is not in known_hosts
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        return client

    def _connect(self, client: paramiko.SSHClient, host: str) -> None:
        kwargs = dict(
            hostname=host,
            port=self.port,
            username=self.username,
            timeout=self.timeout,
        )
        if self.key_file:
            kwargs["key_filename"] = self.key_file
        else:
            kwargs["password"] = self.password
        client.connect(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_sleep(self, host: str) -> bool:
        """Send the sleep command to *host*.

        Returns ``True`` on success, ``False`` on failure.
        """
        client = self._build_client()
        try:
            self._connect(client, host)
            stdin, stdout, stderr = client.exec_command(_SLEEP_COMMAND)
            exit_code = stdout.channel.recv_exit_status()
            if exit_code == 0:
                logger.info("System '%s' has been sent the sleep command.", host)
                return True
            err_output = stderr.read().decode().strip()
            logger.error(
                "Sleep command on '%s' exited with code %d: %s",
                host,
                exit_code,
                err_output,
            )
            return False
        except paramiko.AuthenticationException:
            logger.error("SSH authentication failed for host '%s'.", host)
            return False
        except paramiko.SSHException as exc:
            logger.error("SSH error on host '%s': %s", host, exc)
            return False
        except OSError as exc:
            logger.error("Network error connecting to '%s': %s", host, exc)
            return False
        finally:
            client.close()

    def send_sleep_safe(self, host: str) -> Optional[bool]:
        """Like :meth:`send_sleep` but swallows all exceptions and returns
        ``None`` on unexpected error."""
        try:
            return self.send_sleep(host)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error sending sleep to '%s': %s", host, exc)
            return None
