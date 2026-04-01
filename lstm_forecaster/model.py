"""LSTM model definition for 4-scenario power management forecasting."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Unidirectional stacked LSTM for sequence-to-label classification.

    Args:
        input_size:  Number of input features (default 7 telemetry channels).
        hidden_size: LSTM hidden dimension (default 128).
        num_layers:  Number of stacked LSTM layers (default 2).
        num_classes: Output classes — one per scenario (default 4).
        dropout:     Dropout probability applied *between* LSTM layers (default 0.3).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # dropout is only applied between layers, so set to 0 when num_layers == 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n:      (num_layers, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)

        # Use the last layer's hidden state at the final timestep.
        last_hidden = h_n[-1]  # (batch, hidden_size)
        logits = self.classifier(last_hidden)  # (batch, num_classes)
        return logits


def build_model(config_dict: Dict) -> LSTMForecaster:
    """Factory function that instantiates an :class:`LSTMForecaster` from a config dict.

    Expected keys (with defaults):
      - ``input_size``  (required)
      - ``hidden_size`` (default 128)
      - ``num_layers``  (default 2)
      - ``num_classes`` (default 4)
      - ``dropout``     (default 0.3)
    """
    return LSTMForecaster(
        input_size=config_dict["input_size"],
        hidden_size=config_dict.get("hidden_size", 128),
        num_layers=config_dict.get("num_layers", 2),
        num_classes=config_dict.get("num_classes", 4),
        dropout=config_dict.get("dropout", 0.3),
    )
