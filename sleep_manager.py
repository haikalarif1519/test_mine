"""
Sleep Manager.

Monitors system state using the LSTM model and fuzzy controller, and puts
the system to sleep when the idle condition is confirmed.  Supports both
Windows and Ubuntu (Linux) systems, and can monitor multiple SUT/NUC
hostnames simultaneously with independent per-host state.
"""

import logging
import os
import platform
import subprocess
import time

import joblib
import numpy as np

import config
from data_preprocessing import FEATURE_COLUMNS, RAW_COLUMNS, engineer_features
from fuzzy_logic import build_fuzzy_system, evaluate_sleep_decision
from lstm_model import load_model, predict_state, predict_state_probabilities

logger = logging.getLogger(__name__)


# ── Cross-platform sleep commands ───────────────────────────────────────────

def put_system_to_sleep(hostname=None):
    """
    Put the system identified by *hostname* to sleep.

    When *hostname* is ``None`` or ``"localhost"`` the local machine is
    suspended.  For remote hosts the function logs a warning — remote
    wake-on-LAN / SSH-based suspend should be implemented as needed.

    Raises
    ------
    OSError
        If the operating system is not supported.
    PermissionError
        If the current process lacks the privileges needed to suspend.
    subprocess.CalledProcessError
        If the sleep command fails.
    """
    if hostname is not None and hostname != "localhost":
        logger.warning(
            "Remote sleep for host '%s' is not yet implemented. "
            "Implement SSH-based suspend or wake-on-LAN as needed.",
            hostname,
        )
        return

    system = platform.system()
    logger.info("Putting system to sleep (OS: %s)", system)

    try:
        if system == "Windows":
            subprocess.run(
                ["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"],
                check=True,
            )
        elif system == "Linux":
            subprocess.run(["systemctl", "suspend"], check=True)
        else:
            raise OSError(f"Unsupported OS for sleep: {system}")
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Sleep command failed (exit code %d). "
            "Ensure the process has sufficient privileges (e.g. run as root "
            "or with appropriate polkit policy on Linux).",
            exc.returncode,
        )
        raise PermissionError(
            f"Failed to suspend system: {exc}"
        ) from exc


# ── Live metric collection (stub – replace with Zabbix agent query) ────────

def get_latest_metrics(hostname=None):
    """
    Return the latest metric values as a dict.

    Parameters
    ----------
    hostname : str, optional
        The SUT/NUC hostname to query.  In production this would query the
        Zabbix agent or PostgreSQL database for this host.  The stub
        returns zeros so unit tests and offline runs still work.
    """
    return {col: 0.0 for col in FEATURE_COLUMNS}


# ── Monitoring loop ─────────────────────────────────────────────────────────

class SleepManager:
    """
    Monitors one or more SUT/NUC systems and triggers sleep when the fuzzy
    controller recommends it.

    Each *hostname* has its own independent sequence buffer and consecutive-
    idle counter so that predictions for one host never affect another.

    Parameters
    ----------
    model : lstm_model.SystemStateLSTM
        Trained PyTorch LSTM model.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used during training.
    hostnames : list[str], optional
        List of SUT/NUC hostnames to monitor.  Defaults to
        ``["localhost"]``.
    idle_timeout : int, optional
        Number of consecutive idle predictions required (derived from
        ``config.IDLE_TIMEOUT_MINUTES`` and ``config.POLL_INTERVAL_SECONDS``
        when not given).
    """

    def __init__(self, model, scaler, hostnames=None, idle_timeout=None):
        self.model = model
        self.scaler = scaler
        self.fuzzy_sim = build_fuzzy_system()
        self.hostnames = hostnames or ["localhost"]
        self._host_states = {}

        if idle_timeout is None:
            self.idle_timeout = (
                config.IDLE_TIMEOUT_MINUTES * 60
            ) // config.POLL_INTERVAL_SECONDS
        else:
            self.idle_timeout = idle_timeout

    # ── per-host state ─────────────────────────────────────────────────

    def _get_host_state(self, hostname):
        """Return the mutable state dict for *hostname*, creating it on
        first access."""
        if hostname not in self._host_states:
            self._host_states[hostname] = {
                "sequence_buffer": [],
                "consecutive_idle": 0,
            }
        return self._host_states[hostname]

    def get_consecutive_idle(self, hostname=None):
        """Return the consecutive-idle count for *hostname*."""
        if hostname is None:
            hostname = self.hostnames[0]
        return self._get_host_state(hostname)["consecutive_idle"]

    # Backward-compatible properties that delegate to the first hostname
    # so that existing single-host code and tests keep working.

    @property
    def sequence_buffer(self):
        return self._get_host_state(self.hostnames[0])["sequence_buffer"]

    @sequence_buffer.setter
    def sequence_buffer(self, value):
        self._get_host_state(self.hostnames[0])["sequence_buffer"] = value

    @property
    def consecutive_idle(self):
        return self._get_host_state(self.hostnames[0])["consecutive_idle"]

    @consecutive_idle.setter
    def consecutive_idle(self, value):
        self._get_host_state(self.hostnames[0])["consecutive_idle"] = value

    # ── helpers ─────────────────────────────────────────────────────────

    def _scale_row(self, metric_dict):
        """Scale a single metric dict and return a 1-D numpy array."""
        import pandas as pd
        row_df = pd.DataFrame([metric_dict])[FEATURE_COLUMNS]
        # Replace NaN / inf with 0 to prevent scaler failures
        row_df = row_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        try:
            scaled = self.scaler.transform(row_df.values)
        except ValueError:
            logger.warning("Scaler transform failed – using zeros.")
            scaled = np.zeros((1, len(FEATURE_COLUMNS)))
        return scaled[0]

    def _update_buffer(self, host_state, scaled_row):
        """Append *scaled_row* to the host's buffer, keeping at most
        ``SEQUENCE_LENGTH`` entries."""
        buf = host_state["sequence_buffer"]
        buf.append(scaled_row)
        if len(buf) > config.SEQUENCE_LENGTH:
            host_state["sequence_buffer"] = buf[-config.SEQUENCE_LENGTH:]

    def _buffer_ready(self, host_state):
        return len(host_state["sequence_buffer"]) >= config.SEQUENCE_LENGTH

    # ── main loop ───────────────────────────────────────────────────────

    def step(self, metrics=None, hostname=None):
        """
        Execute a single monitoring step for *hostname*.

        Parameters
        ----------
        metrics : dict, optional
            If *None*, calls :func:`get_latest_metrics`.  The dict should
            contain all ``FEATURE_COLUMNS`` keys (raw + engineered).
            If only raw metric keys are provided, engineered features will
            be computed automatically.
        hostname : str, optional
            The SUT/NUC hostname.  Defaults to the first entry in
            ``self.hostnames``.

        Returns
        -------
        dict with keys ``hostname``, ``state``, ``idle_prob``,
        ``consecutive_idle``, ``should_sleep``, ``sleep_score``.
        Returns ``None`` if the sequence buffer is not yet full.
        """
        if hostname is None:
            hostname = self.hostnames[0]

        host_state = self._get_host_state(hostname)

        if metrics is None:
            metrics = get_latest_metrics(hostname)

        # Auto-engineer features if only raw columns are provided
        if "activity_score" not in metrics:
            import pandas as pd
            row_df = engineer_features(pd.DataFrame([metrics]))
            metrics = row_df.iloc[0].to_dict()

        scaled = self._scale_row(metrics)
        self._update_buffer(host_state, scaled)

        if not self._buffer_ready(host_state):
            logger.info(
                "[%s] Buffer filling: %d / %d",
                hostname,
                len(host_state["sequence_buffer"]),
                config.SEQUENCE_LENGTH,
            )
            return None

        sequence = np.array(host_state["sequence_buffer"], dtype=np.float32)
        state_name, probs = predict_state(self.model, sequence)
        idle_prob = float(probs[config.STATE_LABELS["idle"]])

        if state_name == "idle":
            host_state["consecutive_idle"] += 1
        else:
            host_state["consecutive_idle"] = 0

        should_sleep, sleep_score = evaluate_sleep_decision(
            self.fuzzy_sim, idle_prob, host_state["consecutive_idle"],
        )

        logger.info(
            "[%s] State=%s  idle_prob=%.3f  consecutive_idle=%d  "
            "sleep_score=%.3f  should_sleep=%s",
            hostname, state_name, idle_prob,
            host_state["consecutive_idle"],
            sleep_score, should_sleep,
        )

        return {
            "hostname": hostname,
            "state": state_name,
            "idle_prob": idle_prob,
            "consecutive_idle": host_state["consecutive_idle"],
            "should_sleep": should_sleep,
            "sleep_score": sleep_score,
        }

    def run(self):
        """Run the monitoring loop indefinitely for all hostnames."""
        logger.info(
            "SleepManager started — polling every %ds, idle timeout=%d steps, "
            "monitoring %d host(s): %s",
            config.POLL_INTERVAL_SECONDS,
            self.idle_timeout,
            len(self.hostnames),
            ", ".join(self.hostnames),
        )
        while True:
            for hostname in self.hostnames:
                try:
                    result = self.step(hostname=hostname)
                    if result and result["should_sleep"]:
                        logger.warning(
                            "SLEEP triggered for host '%s'!", hostname,
                        )
                        put_system_to_sleep(hostname)
                except Exception:
                    logger.exception(
                        "Error in monitoring step for host '%s'", hostname,
                    )
            time.sleep(config.POLL_INTERVAL_SECONDS)
