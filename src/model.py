"""
model.py
--------
AI model for idle-state detection.

Architecture
~~~~~~~~~~~~
A :class:`sklearn.ensemble.RandomForestClassifier` is used to classify each
poll-interval sample as *idle* (1) or *active* (0).  When no labelled
training data is available the model is bootstrapped from rule-based labels
produced by :func:`~src.feature_engineering.label_idle`.

Public interface
~~~~~~~~~~~~~~~~
- :class:`IdleDetectionModel`  – wraps training, persistence, and inference.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import (
    FEATURE_COLUMNS,
    label_idle,
    metrics_to_dataframe,
    metrics_to_feature_vector,
)

logger = logging.getLogger(__name__)


class IdleDetectionModel:
    """Random-Forest classifier that predicts whether a system is idle.

    Parameters
    ----------
    model_path:
        File path used to persist / load the trained model.
    idle_threshold:
        Minimum predicted *idle* probability for a sample to be classified as
        idle.  Default 0.7.
    n_estimators:
        Number of trees in the Random-Forest.
    random_state:
        Controls reproducibility.
    """

    def __init__(
        self,
        model_path: str = "sleep_model.joblib",
        idle_threshold: float = 0.7,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.model_path = model_path
        self.idle_threshold = idle_threshold
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self._scaler = StandardScaler()
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Train the classifier on *df*.

        Parameters
        ----------
        df:
            DataFrame with columns matching :data:`~src.feature_engineering.FEATURE_COLUMNS`.
        labels:
            Optional pre-computed binary labels (0 = active, 1 = idle).
            If *None*, rule-based labels are generated automatically using
            :func:`~src.feature_engineering.label_idle`.
        thresholds:
            Keyword arguments forwarded to :func:`~src.feature_engineering.label_idle`
            when *labels* is *None*.

        Returns
        -------
        dict
            Training accuracy and (if a test split was possible) test accuracy.
        """
        if labels is None:
            thresh_kwargs = thresholds or {}
            labels = label_idle(df, **thresh_kwargs)

        X = df[FEATURE_COLUMNS].values
        y = labels.values

        X_scaled = self._scaler.fit_transform(X)

        # Use a train/test split when there are enough samples
        metrics: Dict[str, float] = {}
        if len(X_scaled) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            self._clf.fit(X_train, y_train)
            metrics["train_accuracy"] = float(self._clf.score(X_train, y_train))
            metrics["test_accuracy"] = float(self._clf.score(X_test, y_test))
        else:
            self._clf.fit(X_scaled, y)
            metrics["train_accuracy"] = float(self._clf.score(X_scaled, y))

        self._is_trained = True
        logger.info("Model trained. Metrics: %s", metrics)
        return metrics

    def train_from_metrics_list(
        self,
        metrics_list: List[Dict[str, Any]],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Convenience wrapper – converts a list of raw metrics dicts, then
        calls :meth:`train`."""
        df = metrics_to_dataframe(metrics_list)
        if df.empty:
            raise ValueError("No valid samples found in metrics_list.")
        return self.train(df, thresholds=thresholds)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, metrics: Dict[str, Any]) -> bool:
        """Return ``True`` if the system described by *metrics* is idle.

        Falls back to rule-based detection when the model is not yet trained.
        """
        if not self._is_trained:
            logger.warning("Model not trained – using rule-based fallback.")
            return self._rule_based_idle(metrics)

        feature_vector = metrics_to_feature_vector(metrics)
        if feature_vector is None:
            logger.warning("Incomplete metrics – cannot predict; assuming not idle.")
            return False

        x = self._scaler.transform(feature_vector.reshape(1, -1))
        proba = self._clf.predict_proba(x)[0]

        # predict_proba returns [P(class=0), P(class=1)]
        classes = list(self._clf.classes_)
        idle_class_idx = classes.index(1) if 1 in classes else -1
        if idle_class_idx == -1:
            return False

        idle_prob = proba[idle_class_idx]
        logger.debug("Idle probability: %.3f (threshold=%.3f)", idle_prob, self.idle_threshold)
        return bool(idle_prob >= self.idle_threshold)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the trained model and scaler to :attr:`model_path`."""
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        joblib.dump({"clf": self._clf, "scaler": self._scaler}, self.model_path)
        logger.info("Model saved to '%s'", self.model_path)

    def load(self) -> None:
        """Load a previously saved model from :attr:`model_path`."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file '{self.model_path}' not found. Train the model first."
            )
        data = joblib.load(self.model_path)
        self._clf = data["clf"]
        self._scaler = data["scaler"]
        self._is_trained = True
        logger.info("Model loaded from '%s'", self.model_path)

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_idle(metrics: Dict[str, Any]) -> bool:
        """Simple threshold-based idle check used as a fallback."""
        try:
            return (
                float(metrics.get("cpu_utilization", 100)) < 5.0
                and float(metrics.get("disk_read_bps", float("inf"))) < 102_400
                and float(metrics.get("disk_write_bps", float("inf"))) < 102_400
                and float(metrics.get("network_recv_bps", float("inf"))) < 51_200
                and float(metrics.get("network_sent_bps", float("inf"))) < 51_200
                and float(metrics.get("system_idle_time", 0)) > 95.0
            )
        except (TypeError, ValueError):
            return False
