"""AI model for detecting idle systems using Random Forest classifier."""

import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import config

logger = logging.getLogger(__name__)


class IdleDetectionModel:
    """Random Forest-based classifier for detecting idle systems.

    The model takes system metrics (CPU, memory, disk I/O, network I/O,
    idle time) as input and predicts whether the system is idle or active.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or config.MODEL_PATH
        self.model = None

    def train(self, X, y, test_size=0.2):
        """Train the idle detection model.

        Args:
            X: Feature matrix (numpy array or DataFrame).
            y: Labels (1 = idle, 0 = active).
            test_size: Fraction of data to hold out for evaluation.

        Returns:
            Dictionary with training results (accuracy, report).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y,
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["active", "idle"])

        logger.info("Model training complete. Accuracy: %.4f", accuracy)
        logger.info("Classification Report:\n%s", report)

        return {"accuracy": accuracy, "report": report}

    def predict(self, X):
        """Predict whether the system is idle or active.

        Args:
            X: Feature matrix (numpy array or DataFrame).

        Returns:
            numpy array of predictions (1 = idle, 0 = active).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call train() or load() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get probability estimates for idle/active classification.

        Args:
            X: Feature matrix (numpy array or DataFrame).

        Returns:
            numpy array of shape (n_samples, 2) with probabilities for
            [active, idle].
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call train() or load() first.")
        return self.model.predict_proba(X)

    def is_idle(self, metrics_dict, threshold=None):
        """Determine if a system is idle given its current metrics.

        Args:
            metrics_dict: Dictionary of metric name -> value.
            threshold: Probability threshold for idle classification.
                Defaults to config.IDLE_PROBABILITY_THRESHOLD.

        Returns:
            Tuple of (is_idle: bool, idle_probability: float).
        """
        threshold = threshold or config.IDLE_PROBABILITY_THRESHOLD
        features = np.array([[metrics_dict.get(col, 0.0) for col in config.FEATURE_COLUMNS]])
        proba = self.predict_proba(features)
        idle_prob = float(proba[0][1])  # Probability of idle class
        return bool(idle_prob >= threshold), idle_prob

    def save(self, path=None):
        """Save the trained model to disk.

        Args:
            path: File path. Defaults to self.model_path.
        """
        path = path or self.model_path
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    def load(self, path=None):
        """Load a trained model from disk.

        Args:
            path: File path. Defaults to self.model_path.
        """
        path = path or self.model_path
        self.model = joblib.load(path)
        logger.info("Model loaded from %s", path)

    def get_feature_importance(self):
        """Get feature importance scores from the trained model.

        Returns:
            Dictionary of feature_name -> importance_score.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call train() or load() first.")
        importances = self.model.feature_importances_
        return dict(zip(config.FEATURE_COLUMNS, importances))
