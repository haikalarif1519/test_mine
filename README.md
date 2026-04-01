# test_mine

## LSTM Power Management Scenario Forecaster

A complete LSTM training pipeline that predicts one of four power-management scenarios
for the next 60 seconds based on 120 seconds of system telemetry.

| Label | Scenario | Description |
|-------|----------|-------------|
| 0 | Idle | Low activity — safe to deep-sleep |
| 1 | Power-cycling risk | Rapid wake/sleep — dangerous |
| 2 | Typing / low-power | User active, low compute |
| 3 | Stress / high-power | High compute — do not sleep |

See [`docs/lstm_design.md`](docs/lstm_design.md) for full design details.

---

## Installation

```bash
pip install -r requirements.txt
# or install as an editable package:
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0.

---

## Quick Start

### 1. Preprocess raw data (derive labels if missing)

```bash
python -m lstm_forecaster preprocess testdata/sample.csv output/preprocessed.csv
```

### 2. Train on sample data

```bash
python -m lstm_forecaster train \
    --data-path testdata/sample.csv \
    --window-size 120 \
    --horizon 60 \
    --hidden-size 128 \
    --num-layers 2 \
    --dropout 0.3 \
    --lr 1e-3 \
    --batch-size 64 \
    --max-epochs 50 \
    --patience 5 \
    --checkpoint-dir checkpoints
```

The best checkpoint is saved to `checkpoints/best_model.pt`.

### 3. Evaluate a trained checkpoint

```bash
python -m lstm_forecaster evaluate checkpoints/best_model.pt \
    --data-path testdata/sample.csv \
    --output-dir eval_output \
    --split test
```

Metrics (accuracy, macro F1, Brier score, confusion matrix) are printed and saved to
`eval_output/metrics.json`.

### 4. Export the model

**TorchScript** (default, for PyTorch deployments):

```bash
python -m lstm_forecaster export checkpoints/best_model.pt \
    --output-dir export_output \
    --format torchscript
```

**ONNX** (for cross-runtime deployments):

```bash
python -m lstm_forecaster export checkpoints/best_model.pt \
    --output-dir export_output \
    --format onnx
```

Both formats also write `feature_meta.json` containing feature names, normalization
parameters, and scenario names needed by the inference pipeline.

---

## Python API

```python
from lstm_forecaster.train import train, DEFAULT_CONFIG

config = {
    **DEFAULT_CONFIG,
    "data_path": "testdata/sample.csv",
    "max_epochs": 10,
    "checkpoint_dir": "my_checkpoints",
}
ckpt_path = train(config)
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Layout

```
lstm_forecaster/
  __init__.py      — package version
  dataset.py       — data loading, labeling, normalization, windowing
  model.py         — LSTMForecaster nn.Module + build_model()
  train.py         — training loop, early stopping, checkpointing
  evaluate.py      — metrics computation, metrics.json output
  export.py        — TorchScript and ONNX export
  cli.py           — argparse CLI (preprocess / train / evaluate / export)
  __main__.py      — python -m lstm_forecaster entry point
docs/
  lstm_design.md   — full design document
testdata/
  sample.csv       — 500-row synthetic telemetry (all 4 scenarios)
  sample.jsonl     — first 200 rows as JSONL
tests/
  test_dataset.py  — unit tests for dataset utilities
  test_train.py    — smoke test for the training pipeline
```
