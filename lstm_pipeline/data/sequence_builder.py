import numpy as np
import pandas as pd


def oversample_minority_sequences(
    X: np.ndarray,
    y: np.ndarray,
    minority_classes: tuple = (1, 2, 3),
    copies: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate sequences that contain at least one minority-class target label.

    Oversampling is done at the *sequence* level rather than on raw rows so
    that the temporal structure of each window is preserved intact.  Only the
    training split should be passed here; validation and test sets must not be
    modified.

    Parameters
    ----------
    X : (N, lookback, features) array of input sequences.
    y : (N, predict_steps) array of target labels.
    minority_classes : label values that are considered minority
        (default: Low Load=1, High Load=2, Power Cycle=3).
    copies : how many extra copies of each minority sequence to add.
    """
    minority_mask = np.isin(y, minority_classes).any(axis=1)
    if not minority_mask.any() or copies < 1:
        return X, y
    X_out = np.concatenate([X] + [X[minority_mask]] * copies)
    y_out = np.concatenate([y] + [y[minority_mask]] * copies)
    return X_out, y_out


def create_sequences(
    df: pd.DataFrame, feature_columns: list[str], lookback: int, predict_steps: list[int]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build sliding-window sequences for each device separately.

    Returns a list of ``(X, y)`` tuples, one per device, preserving
    chronological order within each device.  This allows
    ``train_val_test_split`` to split each device's sequences
    independently so that every device is represented in every split.
    """
    if df.empty:
        return []

    max_horizon = max(predict_steps)
    device_sequences = []

    for _, part in df.groupby("device", sort=False):
        part = part.sort_values("datetime").reset_index(drop=True)
        values = part[feature_columns].to_numpy(dtype=float)
        labels = part["label"].to_numpy(dtype=int)

        x_list, y_list = [], []
        for i in range(lookback - 1, len(part) - max_horizon):
            x_start = i - lookback + 1
            x_end = i + 1
            x_list.append(values[x_start:x_end])
            y_list.append([labels[i + step] for step in predict_steps])

        if x_list:
            device_sequences.append(
                (np.asarray(x_list, dtype=float), np.asarray(y_list, dtype=int))
            )

    return device_sequences


def train_val_test_split(
    device_sequences: list[tuple[np.ndarray, np.ndarray]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    """Split each device's sequences chronologically, then concatenate.

    Each device is split independently using the given ratios so that
    train, validation, and test sets each contain data from *every*
    device.  The test set is therefore the most-recent time window for
    every device, providing a true temporal holdout rather than
    exposing the model to entirely unseen devices at evaluation time.

    Parameters
    ----------
    device_sequences:
        List of ``(X, y)`` arrays, one tuple per device, as returned by
        ``create_sequences``.
    train_ratio, val_ratio, test_ratio:
        Proportions that must sum to 1.0.
    """
    if not device_sequences:
        empty = np.empty((0,))
        return (empty, empty), (empty, empty), (empty, empty)

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    X_train_parts, y_train_parts = [], []
    X_val_parts, y_val_parts = [], []
    X_test_parts, y_test_parts = [], []

    for X, y in device_sequences:
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        X_train_parts.append(X[:train_end])
        y_train_parts.append(y[:train_end])
        X_val_parts.append(X[train_end:val_end])
        y_val_parts.append(y[train_end:val_end])
        X_test_parts.append(X[val_end:])
        y_test_parts.append(y[val_end:])

    return (
        (np.concatenate(X_train_parts), np.concatenate(y_train_parts)),
        (np.concatenate(X_val_parts), np.concatenate(y_val_parts)),
        (np.concatenate(X_test_parts), np.concatenate(y_test_parts)),
    )


def guarantee_power_cycle_in_splits(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    power_cycle_label: int = 3,
    min_samples: int = 2,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Ensure Power Cycle sequences appear in the validation and test splits.

    Because Power Cycle events are rare and clustered near the start of each
    device's recording window, a strict chronological split can leave val/test
    with zero Power Cycle examples.  This makes it impossible to measure how
    well the model detects that class during training and evaluation.

    This function *copies* up to ``min_samples`` Power Cycle sequences from
    the training split into any val/test split that lacks them.  The sequences
    are copied (not moved) so that the already-tiny Power Cycle representation
    in the training set is not further reduced.

    Parameters
    ----------
    train, val, test : (X, y) array pairs as returned by ``train_val_test_split``.
    power_cycle_label : integer label used for the Power Cycle class (default 3).
    min_samples : maximum number of Power Cycle sequences to copy into each
        split that is missing them (default 2).

    Returns
    -------
    Updated (train, val, test) tuples.  train is returned unchanged; val and
    test may have extra Power Cycle sequences appended.
    """
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    if y_train.ndim == 1:
        pc_mask = y_train == power_cycle_label
    else:
        pc_mask = np.isin(y_train, power_cycle_label).any(axis=1)

    pc_indices = np.where(pc_mask)[0]
    if len(pc_indices) == 0:
        return train, val, test

    n_to_copy = min(min_samples, len(pc_indices))
    chosen = pc_indices[:n_to_copy]

    def _has_power_cycle(y: np.ndarray) -> bool:
        if y.size == 0:
            return False
        if y.ndim == 1:
            return bool((y == power_cycle_label).any())
        return bool(np.isin(y, power_cycle_label).any())

    if not _has_power_cycle(y_val):
        X_val = np.concatenate([X_val, X_train[chosen]])
        y_val = np.concatenate([y_val, y_train[chosen]])

    if not _has_power_cycle(y_test):
        X_test = np.concatenate([X_test, X_train[chosen]])
        y_test = np.concatenate([y_test, y_train[chosen]])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
