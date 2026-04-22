import numpy as np
import pandas as pd


def create_sequences(df: pd.DataFrame, feature_columns: list[str], lookback: int, predict_steps: list[int]):
    if df.empty:
        return np.empty((0, lookback, len(feature_columns))), np.empty((0, len(predict_steps)))

    x_list, y_list = [], []
    max_horizon = max(predict_steps)

    for _, part in df.groupby("device", sort=False):
        part = part.sort_values("datetime").reset_index(drop=True)
        values = part[feature_columns].to_numpy(dtype=float)
        labels = part["label"].to_numpy(dtype=int)
        for i in range(lookback - 1, len(part) - max_horizon):
            x_start = i - lookback + 1
            x_end = i + 1
            x_list.append(values[x_start:x_end])
            y_list.append([labels[i + step] for step in predict_steps])

    return np.asarray(x_list, dtype=float), np.asarray(y_list, dtype=int)


def train_val_test_split(X, y, train_ratio: float, val_ratio: float, test_ratio: float):
    total = len(X)
    if total == 0:
        return (X, y), (X, y), (X, y)

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return (
        (X[:train_end], y[:train_end]),
        (X[train_end:val_end], y[train_end:val_end]),
        (X[val_end:], y[val_end:]),
    )
