import torch
import torch.nn as nn


class EncoderDecoderLSTM(nn.Module):
    """Encoder-decoder LSTM for multi-step sequence classification.

    Architecture mirrors the original Keras model:
      Encoder: LSTM(layer1) → LSTM(layer2) → Dropout → RepeatVector
      Decoder: LSTM(layer2) → Linear(output_classes)

    Forward returns raw logits of shape (N, output_steps, output_classes).
    Apply softmax externally when probabilities are needed.
    """

    def __init__(
        self,
        feature_count: int,
        lstm_units_layer1: int,
        lstm_units_layer2: int,
        dropout: float,
        output_steps: int,
        output_classes: int,
    ):
        super().__init__()
        self.output_steps = output_steps
        self.encoder_lstm1 = nn.LSTM(feature_count, lstm_units_layer1, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(lstm_units_layer1, lstm_units_layer2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.decoder_lstm = nn.LSTM(lstm_units_layer2, lstm_units_layer2, batch_first=True)
        self.output_layer = nn.Linear(lstm_units_layer2, output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x, _ = self.encoder_lstm1(x)                          # (N, T, units1)
        _, (h, _) = self.encoder_lstm2(x)                     # h: (1, N, units2)
        x = self.dropout(h.squeeze(0))                        # (N, units2)

        # RepeatVector
        x = x.unsqueeze(1).repeat(1, self.output_steps, 1)   # (N, output_steps, units2)

        # Decoder
        x, _ = self.decoder_lstm(x)                           # (N, output_steps, units2)
        return self.output_layer(x)                           # (N, output_steps, output_classes)


def build_lstm_model(config: dict) -> EncoderDecoderLSTM:
    mdl_cfg = config["model"]
    feature_count = len(config["feature_columns"])
    return EncoderDecoderLSTM(
        feature_count=feature_count,
        lstm_units_layer1=mdl_cfg["lstm_units_layer1"],
        lstm_units_layer2=mdl_cfg["lstm_units_layer2"],
        dropout=mdl_cfg["dropout"],
        output_steps=mdl_cfg["output_steps"],
        output_classes=mdl_cfg["output_classes"],
    )
