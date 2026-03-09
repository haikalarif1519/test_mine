"""
LSTM Model for System State Forecasting.

Builds, trains and uses a multi-class LSTM model that predicts the next
system state (Idle / Low Load / High Load) from a sequence of historical
metric observations.
"""

import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config

logger = logging.getLogger(__name__)


def build_model(
    seq_length=None,
    num_features=None,
    num_classes=None,
    lstm_units=None,
    dropout_rate=None,
):
    """
    Build and compile a two-layer LSTM classifier.

    Returns a compiled :class:`keras.Model`.
    """
    seq_length = seq_length or config.SEQUENCE_LENGTH
    num_features = num_features or config.NUM_FEATURES
    num_classes = num_classes or config.NUM_CLASSES
    lstm_units = lstm_units or config.LSTM_UNITS
    dropout_rate = dropout_rate or config.DROPOUT_RATE

    model = keras.Sequential([
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            input_shape=(seq_length, num_features),
        ),
        layers.Dropout(dropout_rate),
        layers.LSTM(lstm_units),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train, y_train, epochs=None, batch_size=None,
                validation_split=None):
    """
    Train *model* and return the training history.
    """
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    validation_split = validation_split if validation_split is not None else config.VALIDATION_SPLIT

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


def save_model(model, path=None):
    """Save the trained model to disk."""
    path = path or config.MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info("Model saved to %s", path)


def load_model(path=None):
    """Load a trained model from disk."""
    path = path or config.MODEL_PATH
    model = keras.models.load_model(path)
    logger.info("Model loaded from %s", path)
    return model


def predict_state_probabilities(model, sequence):
    """
    Given a single input sequence of shape ``(seq_length, num_features)``,
    return the predicted class probabilities as a 1-D numpy array of length
    ``num_classes`` (idle, low_load, high_load).
    """
    # Add batch dimension
    if sequence.ndim == 2:
        sequence = np.expand_dims(sequence, axis=0)
    probabilities = model.predict(sequence, verbose=0)
    return probabilities[0]


def predict_state(model, sequence):
    """
    Predict the most likely state for a single input sequence.

    Returns
    -------
    state_name : str
        ``"idle"``, ``"low_load"`` or ``"high_load"``.
    probabilities : np.ndarray
        Class probabilities.
    """
    probs = predict_state_probabilities(model, sequence)
    predicted_class = int(np.argmax(probs))
    return config.STATE_NAMES[predicted_class], probs
