"""
LSTM Model for System State Forecasting (PyTorch).

Builds, trains and uses a multi-class LSTM model that predicts the next
system state (Idle / Low Load / High Load) from a sequence of historical
metric observations.  Uses weighted CrossEntropyLoss to handle class
imbalance.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config

logger = logging.getLogger(__name__)


class SystemStateLSTM(nn.Module):
    """Two-layer LSTM classifier for system state prediction."""

    def __init__(self, num_features=None, lstm_units=None, num_classes=None,
                 dropout_rate=None):
        super().__init__()
        num_features = num_features or config.NUM_FEATURES
        lstm_units = lstm_units or config.LSTM_UNITS
        num_classes = num_classes or config.NUM_CLASSES
        dropout_rate = dropout_rate or config.DROPOUT_RATE

        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(
            input_size=lstm_units,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # take last time-step
        out = self.fc(out)
        return out


def build_model(num_features=None, lstm_units=None, num_classes=None,
                dropout_rate=None):
    """
    Build a :class:`SystemStateLSTM` model.

    Returns an untrained model instance.
    """
    return SystemStateLSTM(
        num_features=num_features,
        lstm_units=lstm_units,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


def train_model(model, X_train, y_train, class_weights=None, epochs=None,
                batch_size=None, validation_split=None, learning_rate=None):
    """
    Train *model* using weighted CrossEntropyLoss to handle class imbalance.

    Parameters
    ----------
    model : SystemStateLSTM
    X_train : np.ndarray of shape ``(n, seq_len, features)``
    y_train : np.ndarray of shape ``(n,)``
    class_weights : np.ndarray, optional
        Per-class weights.  When *None*, all classes are weighted equally.
    epochs : int
    batch_size : int
    validation_split : float
    learning_rate : float

    Returns
    -------
    history : dict  with ``"train_loss"`` and ``"val_loss"`` lists.
    """
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    learning_rate = learning_rate or config.LEARNING_RATE
    validation_split = (
        validation_split if validation_split is not None
        else config.VALIDATION_SPLIT
    )

    # Convert to tensors
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)

    # Train / validation split
    n_val = int(len(X) * validation_split)
    n_train = len(X) - n_val
    X_tr, X_val = X[:n_train], X[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
    )

    # Weighted loss
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    else:
        weight_tensor = None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_batch)

        train_loss = running_loss / n_train

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
        model.train()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch %d/%d – train_loss=%.4f  val_loss=%.4f",
            epoch + 1, epochs, train_loss, val_loss,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def save_model(model, path=None):
    """Save the trained model to disk."""
    path = path or config.MODEL_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Model saved to %s", path)


def load_model(path=None, **kwargs):
    """Load a trained model from disk."""
    path = path or config.MODEL_PATH
    model = SystemStateLSTM(**kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    logger.info("Model loaded from %s", path)
    return model


def predict_state_probabilities(model, sequence):
    """
    Given a single input sequence of shape ``(seq_length, num_features)``,
    return the predicted class probabilities as a 1-D numpy array of length
    ``num_classes`` (idle, low_load, high_load).
    """
    model.eval()
    if isinstance(sequence, np.ndarray):
        sequence = torch.tensor(sequence, dtype=torch.float32)
    # Add batch dimension
    if sequence.dim() == 2:
        sequence = sequence.unsqueeze(0)
    with torch.no_grad():
        logits = model(sequence)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities[0].numpy()


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
