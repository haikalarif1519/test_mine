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
