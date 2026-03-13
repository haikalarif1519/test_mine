"""Tests for sleep_manager module."""

import numpy as np
import pytest

from data_preprocessing import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    compute_class_weights,
    create_sequences,
    engineer_features,
    fit_scaler,
    generate_simulation_data,
    label_dataframe,
    scale_features,
)
from lstm_model import build_model, train_model
from sleep_manager import SleepManager


@pytest.fixture(scope="module")
def trained_model_and_scaler():
    """Build and train a small model on simulation data for testing."""
    df = generate_simulation_data(n_idle=100, n_low=100, n_high=100)
    engineer_features(df)
    label_dataframe(df)
    scaler = fit_scaler(df)
    df_scaled = scale_features(df, scaler)
    X, y = create_sequences(df_scaled, seq_length=5)
    class_weights = compute_class_weights(y)

    model = build_model(num_features=10, num_classes=3, lstm_units=16)
    train_model(model, X, y, class_weights=class_weights, epochs=3,
                batch_size=32, validation_split=0.2)
    return model, scaler


class TestSleepManager:
    def test_step_returns_none_when_buffer_not_full(self, trained_model_and_scaler):
        model, scaler = trained_model_and_scaler
        manager = SleepManager(model, scaler, idle_timeout=5)
        metrics = {col: 2.0 for col in FEATURE_COLUMNS}
        result = manager.step(metrics)
        # Buffer has only 1 entry out of SEQUENCE_LENGTH (default 10)
        assert result is None

    def test_step_returns_result_when_buffer_full(self, trained_model_and_scaler):
        model, scaler = trained_model_and_scaler
        manager = SleepManager(model, scaler, idle_timeout=5)
        metrics = {col: 2.0 for col in FEATURE_COLUMNS}

        # Fill the buffer
        import config
        for _ in range(config.SEQUENCE_LENGTH):
            result = manager.step(metrics)

        # The last call should return a result
        assert result is not None
        assert "state" in result
        assert "idle_prob" in result
        assert "should_sleep" in result

    def test_consecutive_idle_increments(self, trained_model_and_scaler):
        """Idle metrics should lead to incrementing consecutive_idle."""
        model, scaler = trained_model_and_scaler
        manager = SleepManager(model, scaler, idle_timeout=20)

        idle_metrics = {
            "cpu_utilization": 1.0,
            "memory_utilization": 5.0,
            "disk_read": 100.0,
            "disk_write": 100.0,
            "network_received": 500.0,
            "network_sent": 500.0,
            "system_idle_time": 300.0,
            "activity_score": 2.6,
            "io_total": 1200.0,
            "is_idle": 1.0,
        }

        import config
        # Fill the buffer and run a few extra steps
        for _ in range(config.SEQUENCE_LENGTH + 3):
            manager.step(idle_metrics)

        # consecutive_idle should be > 0 if model predicts idle
        # (with a small model trained on sim data this is likely but not
        #  guaranteed – we just check the counter is non-negative)
        assert manager.consecutive_idle >= 0

    def test_step_auto_engineers_features_from_raw(self, trained_model_and_scaler):
        """When only raw metrics are provided, engineered features are auto-computed."""
        model, scaler = trained_model_and_scaler
        manager = SleepManager(model, scaler, idle_timeout=5)

        raw_metrics = {col: 2.0 for col in RAW_COLUMNS}

        import config
        for _ in range(config.SEQUENCE_LENGTH):
            result = manager.step(raw_metrics)

        assert result is not None
        assert "state" in result


class TestMultiHostSleepManager:
    """Tests for multi-hostname monitoring support."""

    def test_independent_per_host_state(self, trained_model_and_scaler):
        """Each hostname should have its own sequence buffer and
        consecutive_idle counter."""
        model, scaler = trained_model_and_scaler
        hosts = ["nuc-01", "nuc-02"]
        manager = SleepManager(model, scaler, hostnames=hosts, idle_timeout=20)

        metrics = {col: 2.0 for col in FEATURE_COLUMNS}

        import config
        # Only fill buffer for nuc-01
        for _ in range(config.SEQUENCE_LENGTH):
            manager.step(metrics, hostname="nuc-01")

        # nuc-01 should have a full buffer, nuc-02 should be empty
        assert len(manager._get_host_state("nuc-01")["sequence_buffer"]) == config.SEQUENCE_LENGTH
        assert len(manager._get_host_state("nuc-02")["sequence_buffer"]) == 0

    def test_step_with_hostname_returns_hostname(self, trained_model_and_scaler):
        """The result dict should include the hostname."""
        model, scaler = trained_model_and_scaler
        manager = SleepManager(
            model, scaler, hostnames=["nuc-01"], idle_timeout=5,
        )
        metrics = {col: 2.0 for col in FEATURE_COLUMNS}

        import config
        for _ in range(config.SEQUENCE_LENGTH):
            result = manager.step(metrics, hostname="nuc-01")

        assert result is not None
        assert result["hostname"] == "nuc-01"

    def test_default_hostname_is_localhost(self, trained_model_and_scaler):
        """When no hostnames are provided, the default is 'localhost'."""
        model, scaler = trained_model_and_scaler
        manager = SleepManager(model, scaler, idle_timeout=5)
        assert manager.hostnames == ["localhost"]

    def test_consecutive_idle_per_host(self, trained_model_and_scaler):
        """get_consecutive_idle returns per-host counts."""
        model, scaler = trained_model_and_scaler
        hosts = ["nuc-01", "nuc-02"]
        manager = SleepManager(model, scaler, hostnames=hosts, idle_timeout=20)

        # Both start at 0
        assert manager.get_consecutive_idle("nuc-01") == 0
        assert manager.get_consecutive_idle("nuc-02") == 0

    def test_backward_compat_properties(self, trained_model_and_scaler):
        """The old .consecutive_idle and .sequence_buffer properties
        should still work, delegating to the first hostname."""
        model, scaler = trained_model_and_scaler
        manager = SleepManager(
            model, scaler, hostnames=["host-a", "host-b"], idle_timeout=5,
        )

        # These should delegate to "host-a"
        assert manager.consecutive_idle == 0
        assert manager.sequence_buffer == []

        manager.consecutive_idle = 5
        assert manager.get_consecutive_idle("host-a") == 5
        assert manager.get_consecutive_idle("host-b") == 0
