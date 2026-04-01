"""Model export utilities — TorchScript and ONNX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from lstm_forecaster.dataset import FEATURE_COLS, SCENARIO_NAMES
from lstm_forecaster.evaluate import load_checkpoint
from lstm_forecaster.model import LSTMForecaster


def _build_feature_meta(cfg: Dict, normalizer_dict: Dict, feature_cols: list) -> Dict:
    """Assemble the ``feature_meta.json`` payload."""
    normalizer_params = {
        col: {
            "mean": normalizer_dict["mean"].get(col, 0.0),
            "std": normalizer_dict["std"].get(col, 1.0),
        }
        for col in feature_cols
    }
    return {
        "feature_cols": feature_cols,
        "window_size": cfg.get("window_size", 120),
        "horizon": cfg.get("horizon", 60),
        "num_classes": cfg.get("num_classes", 4),
        "scenario_names": SCENARIO_NAMES,
        "normalizer_params": normalizer_params,
    }


def export_torchscript(checkpoint_path: str | Path, output_dir: str | Path) -> Path:
    """Export the trained model to TorchScript (trace-based).

    Saves:
      - ``lstm_forecaster.pt``   — TorchScript traced model
      - ``feature_meta.json``    — Inference metadata

    Args:
        checkpoint_path: Path to the training checkpoint ``.pt`` file.
        output_dir:      Directory where artifacts will be written.

    Returns:
        Path to the exported ``lstm_forecaster.pt``.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint["config"]
    feature_cols = checkpoint.get("feature_cols", FEATURE_COLS)

    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        num_classes=cfg.get("num_classes", 4),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trace with a dummy input (batch=1, seq_len=window_size, features)
    window_size = cfg.get("window_size", 120)
    dummy_input = torch.zeros(1, window_size, len(feature_cols))

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

    model_path = output_dir / "lstm_forecaster.pt"
    traced.save(str(model_path))

    # Save metadata
    meta = _build_feature_meta(cfg, checkpoint["normalizer"], feature_cols)
    meta_path = output_dir / "feature_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"TorchScript model saved to : {model_path}")
    print(f"Feature metadata saved to  : {meta_path}")
    return model_path


def export_onnx(checkpoint_path: str | Path, output_dir: str | Path) -> Path:
    """Export the trained model to ONNX format.

    Saves:
      - ``lstm_forecaster.onnx`` — ONNX model
      - ``feature_meta.json``    — Inference metadata

    If the ``onnx`` package is not installed, this function prints a warning
    and returns without raising an exception.

    Args:
        checkpoint_path: Path to the training checkpoint ``.pt`` file.
        output_dir:      Directory where artifacts will be written.

    Returns:
        Path to the exported ``lstm_forecaster.onnx``.
    """
    try:
        import onnx  # noqa: F401 — validate import
    except ImportError:
        print("WARNING: 'onnx' is not installed. Skipping ONNX export.")
        return Path(output_dir) / "lstm_forecaster.onnx"

    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint["config"]
    feature_cols = checkpoint.get("feature_cols", FEATURE_COLS)

    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        num_classes=cfg.get("num_classes", 4),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_size = cfg.get("window_size", 120)
    dummy_input = torch.zeros(1, window_size, len(feature_cols))

    onnx_path = output_dir / "lstm_forecaster.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["telemetry_window"],
        output_names=["logits"],
        dynamic_axes={
            "telemetry_window": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Save metadata
    meta = _build_feature_meta(cfg, checkpoint["normalizer"], feature_cols)
    meta_path = output_dir / "feature_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"ONNX model saved to       : {onnx_path}")
    print(f"Feature metadata saved to : {meta_path}")
    return onnx_path
