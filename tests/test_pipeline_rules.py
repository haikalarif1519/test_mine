import unittest

import numpy as np
import pandas as pd

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
        X, y = create_sequences(df, ["f1", "f2"], lookback=30, predict_steps=[5, 10, 15])
        self.assertEqual(X.shape, (60 - 15 - 29, 30, 2))
        self.assertEqual(y.shape, (60 - 15 - 29, 3))
        self.assertListEqual(y[0].tolist(), [df.loc[34, "label"], df.loc[39, "label"], df.loc[44, "label"]])

    def test_split_no_shuffle_ordered(self):
        X = np.arange(100).reshape(20, 5)
        y = np.arange(20)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y, 0.5, 0.25, 0.25)
        self.assertTrue(np.array_equal(y_train, np.arange(10)))
        self.assertTrue(np.array_equal(y_val, np.arange(10, 15)))
        self.assertTrue(np.array_equal(y_test, np.arange(15, 20)))


if __name__ == "__main__":
    unittest.main()
