"""
Sleep Manager.

Monitors system state using the LSTM model and fuzzy controller, and puts
the system to sleep when the idle condition is confirmed.  Supports both
Windows and Ubuntu (Linux) systems.
"""

import logging
import os
import platform
import subprocess
import time

import joblib
import numpy as np

import config
from data_preprocessing import FEATURE_COLUMNS, scale_features
from fuzzy_logic import build_fuzzy_system, evaluate_sleep_decision
from lstm_model import load_model, predict_state, predict_state_probabilities

logger = logging.getLogger(__name__)


# ── Cross-platform sleep commands ───────────────────────────────────────────

def put_system_to_sleep():
    """
    Put the local system to sleep using the appropriate OS command.

    Raises
    ------
    OSError
        If the operating system is not supported.
    PermissionError
        If the current process lacks the privileges needed to suspend.
    subprocess.CalledProcessError
        If the sleep command fails.
    """
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

def get_latest_metrics():
    """
    Return the latest metric values as a dict.

    In production this would query the Zabbix agent or the PostgreSQL
    database.  The stub returns a dict of zeros so unit tests and offline
    runs can still exercise the pipeline.
    """
    return {col: 0.0 for col in FEATURE_COLUMNS}


# ── Monitoring loop ─────────────────────────────────────────────────────────

class SleepManager:
    """
    Continuously monitors system state and triggers sleep when the fuzzy
    controller recommends it.

    Parameters
    ----------
    model : keras.Model
        Trained LSTM model.
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler used during training.
    idle_timeout : int
        Number of consecutive idle predictions required (derived from
        ``config.IDLE_TIMEOUT_MINUTES`` and ``config.POLL_INTERVAL_SECONDS``).
    """

    def __init__(self, model, scaler, idle_timeout=None):
        self.model = model
        self.scaler = scaler
        self.fuzzy_sim = build_fuzzy_system()
        self.sequence_buffer = []
        self.consecutive_idle = 0

        if idle_timeout is None:
            self.idle_timeout = (
                config.IDLE_TIMEOUT_MINUTES * 60
            ) // config.POLL_INTERVAL_SECONDS
        else:
            self.idle_timeout = idle_timeout

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

    def _update_buffer(self, scaled_row):
        """Append *scaled_row* and keep only the last ``SEQUENCE_LENGTH`` entries."""
        self.sequence_buffer.append(scaled_row)
        if len(self.sequence_buffer) > config.SEQUENCE_LENGTH:
            self.sequence_buffer = self.sequence_buffer[-config.SEQUENCE_LENGTH:]

    def _buffer_ready(self):
        return len(self.sequence_buffer) >= config.SEQUENCE_LENGTH

    # ── main loop ───────────────────────────────────────────────────────

    def step(self, metrics=None):
        """
        Execute a single monitoring step.

        Parameters
        ----------
        metrics : dict, optional
            If *None*, calls :func:`get_latest_metrics`.

        Returns
        -------
        dict with keys ``state``, ``idle_prob``, ``consecutive_idle``,
        ``should_sleep``, ``sleep_score``.  Returns ``None`` if the
        sequence buffer is not yet full.
        """
        if metrics is None:
            metrics = get_latest_metrics()

        scaled = self._scale_row(metrics)
        self._update_buffer(scaled)

        if not self._buffer_ready():
            logger.info(
                "Buffer filling: %d / %d",
                len(self.sequence_buffer), config.SEQUENCE_LENGTH,
            )
            return None

        sequence = np.array(self.sequence_buffer)
        state_name, probs = predict_state(self.model, sequence)
        idle_prob = float(probs[config.STATE_LABELS["idle"]])

        if state_name == "idle":
            self.consecutive_idle += 1
        else:
            self.consecutive_idle = 0

        should_sleep, sleep_score = evaluate_sleep_decision(
            self.fuzzy_sim, idle_prob, self.consecutive_idle,
        )

        logger.info(
            "State=%s  idle_prob=%.3f  consecutive_idle=%d  "
            "sleep_score=%.3f  should_sleep=%s",
            state_name, idle_prob, self.consecutive_idle,
            sleep_score, should_sleep,
        )

        return {
            "state": state_name,
            "idle_prob": idle_prob,
            "consecutive_idle": self.consecutive_idle,
            "should_sleep": should_sleep,
            "sleep_score": sleep_score,
        }

    def run(self):
        """Run the monitoring loop indefinitely."""
        logger.info(
            "SleepManager started — polling every %ds, idle timeout=%d steps",
            config.POLL_INTERVAL_SECONDS, self.idle_timeout,
        )
        while True:
            try:
                result = self.step()
                if result and result["should_sleep"]:
                    logger.warning("SLEEP triggered!")
                    put_system_to_sleep()
            except Exception:
                logger.exception("Error in monitoring step")
            time.sleep(config.POLL_INTERVAL_SECONDS)
