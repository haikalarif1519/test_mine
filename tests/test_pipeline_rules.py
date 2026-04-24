import unittest

import numpy as np
import pandas as pd

from lstm_pipeline.data.preprocessor import Preprocessor
from lstm_pipeline.data.sequence_builder import create_sequences, train_val_test_split
from lstm_pipeline.features.feature_engineer import add_time_features


class PipelineRuleTests(unittest.TestCase):
    def test_time_features_values(self):
        df = pd.DataFrame(
            {
                "date": ["2026-02-02", "2026-02-08"],  # Monday, Sunday
                "record_time": ["09:00:00", "20:00:00"],
            }
        )
        out = add_time_features(df)
        self.assertListEqual(out["hour"].tolist(), [9, 20])
        self.assertListEqual(out["day_of_week"].tolist(), [0, 6])
        self.assertListEqual(out["is_weekend"].tolist(), [0, 1])
        self.assertListEqual(out["is_working_hour"].tolist(), [1, 0])

    def test_sequence_shape_and_targets(self):
        rows = 60
        df = pd.DataFrame(
            {
                "device": ["SUT1"] * rows,
                "datetime": pd.date_range("2026-02-01", periods=rows, freq="min"),
                "f1": np.arange(rows),
                "f2": np.arange(rows) + 100,
                "label": np.array([i % 3 for i in range(rows)]),
            }
        )
        sequences = create_sequences(df, ["f1", "f2"], lookback=30, predict_steps=[5, 10, 15])
        self.assertEqual(len(sequences), 1)
        X, y = sequences[0]
        self.assertEqual(X.shape, (60 - 15 - 29, 30, 2))
        self.assertEqual(y.shape, (60 - 15 - 29, 3))
        self.assertListEqual(y[0].tolist(), [df.loc[34, "label"], df.loc[39, "label"], df.loc[44, "label"]])

    def test_split_no_shuffle_ordered(self):
        X = np.arange(100).reshape(20, 5)
        y = np.arange(20)
        device_sequences = [(X, y)]
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
            device_sequences, 0.5, 0.25, 0.25
        )
        self.assertTrue(np.array_equal(y_train, np.arange(10)))
        self.assertTrue(np.array_equal(y_val, np.arange(10, 15)))
        self.assertTrue(np.array_equal(y_test, np.arange(15, 20)))

    def test_split_per_device_all_devices_represented(self):
        """Every device must appear in train, val, and test after per-device split."""
        X1 = np.zeros((20, 5))
        y1 = np.zeros(20, dtype=float)
        X2 = np.ones((20, 5))
        y2 = np.ones(20, dtype=float)
        device_sequences = [(X1, y1), (X2, y2)]
        (_, y_train), (_, y_val), (_, y_test) = train_val_test_split(
            device_sequences, 0.7, 0.15, 0.15
        )
        # Both label values (0.0 for device 1, 1.0 for device 2) must appear in every split
        for split_labels, split_name in [(y_train, "train"), (y_val, "val"), (y_test, "test")]:
            self.assertIn(0.0, split_labels, msg=f"Device 1 missing from {split_name}")
            self.assertIn(1.0, split_labels, msg=f"Device 2 missing from {split_name}")

    def test_power_cycle_injected_for_long_gaps(self):
        """Rows synthesised inside a gap ≥ power_cycle_gap_minutes get systemstate='Power Cycle'
        and zeroed hardware metrics; rows outside the gap keep their original state."""
        config = {
            "data": {"power_cycle_gap_minutes": 30},
            "label_map": {"Idle": 0, "Low Load": 1, "High Load": 2, "Power Cycle": 3},
            "feature_columns": ["cpu_utilization", "idle_time"],
            "scale_columns": [],
            "paths": {"scaler": "/tmp/test_scaler.pkl"},
        }
        preprocessor = Preprocessor(config)

        # Two real observations separated by a 60-minute gap (well above the 30-min threshold).
        t0 = pd.Timestamp("2026-02-01 08:00:00")
        t1 = pd.Timestamp("2026-02-01 09:00:00")  # 60-min gap after t0
        df = pd.DataFrame(
            {
                "device": ["SUT1", "SUT1"],
                "datetime": [t0, t1],
                "date": [t0.date(), t1.date()],
                "systemstate": ["Idle", "Idle"],
                "cpu_utilization": [5.0, 5.0],
                "idle_time": [90.0, 90.0],
            }
        )

        result = preprocessor.handle_missing(df)

        # Rows strictly inside the gap (08:01 – 08:59) must be Power Cycle.
        gap_rows = result[(result["datetime"] > t0) & (result["datetime"] < t1)]
        self.assertGreater(len(gap_rows), 0, "Expected synthetic gap rows to be created")
        self.assertTrue(
            (gap_rows["systemstate"] == "Power Cycle").all(),
            "All gap rows should be labelled Power Cycle",
        )
        self.assertTrue((gap_rows["cpu_utilization"] == 0.0).all(), "cpu_utilization should be zeroed in gap")
        self.assertTrue((gap_rows["idle_time"] == 0.0).all(), "idle_time should be zeroed in gap")

        # The two real boundary rows must keep their original state.
        real_rows = result[result["datetime"].isin([t0, t1])]
        self.assertTrue(
            (real_rows["systemstate"] == "Idle").all(),
            "Boundary rows should retain their original systemstate",
        )

    def test_short_gap_not_labelled_power_cycle(self):
        """A gap shorter than the threshold must not introduce Power Cycle rows."""
        config = {
            "data": {"power_cycle_gap_minutes": 30},
            "label_map": {"Idle": 0, "Low Load": 1, "High Load": 2, "Power Cycle": 3},
            "feature_columns": ["cpu_utilization"],
            "scale_columns": [],
            "paths": {"scaler": "/tmp/test_scaler.pkl"},
        }
        preprocessor = Preprocessor(config)

        t0 = pd.Timestamp("2026-02-01 08:00:00")
        t1 = pd.Timestamp("2026-02-01 08:10:00")  # only 10-min gap — below threshold
        df = pd.DataFrame(
            {
                "device": ["SUT1", "SUT1"],
                "datetime": [t0, t1],
                "date": [t0.date(), t1.date()],
                "systemstate": ["Idle", "Idle"],
                "cpu_utilization": [5.0, 5.0],
            }
        )

        result = preprocessor.handle_missing(df)
        self.assertFalse(
            (result["systemstate"] == "Power Cycle").any(),
            "Short gap should not produce Power Cycle rows",
        )


if __name__ == "__main__":
    unittest.main()
