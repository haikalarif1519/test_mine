# LSTM Power Management Scenario Forecaster — Design Document

## 1. Problem Formulation

### Task Type: Sequence-to-Label Classification

Given a lookback window of **120 seconds** of multivariate telemetry, the model predicts the **system scenario at t+60 seconds** (a single label).

**Why sequence-to-label?**  
Power-management decisions are inherently temporal — a spike in CPU utilization that lasts 1 second is harmless, but one that persists for 30 seconds signals a stress state. A fixed lookback window captures short-term trends, oscillations, and transitions while keeping inference latency low.

**Why t+60 prediction horizon?**  
60 seconds gives the power-management controller enough lead time to pre-warm or pre-cool subsystems, schedule deep-sleep transitions, or allocate power budgets — without being so far ahead that the prediction becomes noise.

### Output Classes

| Label | Name | Description |
|-------|------|-------------|
| 0 | Idle | Low activity — safe to enter deep sleep |
| 1 | Power-cycling risk | Rapid wake/sleep oscillations — dangerous |
| 2 | Typing / low-power | User active, low compute — stay awake, low power |
| 3 | Stress / high-power | High compute — do not sleep, full power budget |

---

## 2. Feature Schema and Normalization

### Input Features (7 channels)

| Feature | Unit | Description |
|---------|------|-------------|
| `cpu_utilization` | % (0–100) | Aggregate CPU usage across all cores |
| `power_draw_watts` | W | Total system power draw |
| `keyboard_interrupt_rate` | interrupts/s | Keyboard activity proxy |
| `mouse_interrupt_rate` | interrupts/s | Mouse activity proxy |
| `memory_utilization` | % (0–100) | RAM utilization |
| `disk_io_mbps` | MB/s | Aggregate disk I/O throughput |
| `network_mbps` | Mb/s | Network throughput (in + out) |

### Normalization Strategy: Z-Score (Standardization)

Each feature is normalized to zero mean and unit variance using statistics **computed on the training set only**, then applied to val and test sets.

**Formula:**  
```
x_norm = (x - μ_train) / (σ_train + ε)
```
where `ε = 1e-8` to avoid division by zero.

**Rationale:**  
- Z-score is preferred over min-max here because power draw and interrupt rates can spike well beyond historical maxima during anomalous events. Z-score is more robust to outliers.  
- Normalizer parameters (mean, std) are saved alongside the model for consistent inference.

---

## 3. Labeling Strategy

Labels are derived from raw telemetry using threshold rules (priority order):

```
if cpu_utilization > 70 OR power_draw_watts > 30:
    label = 3  # Stress/high-power
elif keyboard_interrupt_rate > 5 OR mouse_interrupt_rate > 5:
    label = 2  # Typing/low-power
elif cpu_utilization < 10 AND power_draw_watts < 10:
    label = 0  # Idle
else:
    label = 1  # Power-cycling risk
```

These thresholds are domain-derived and can be calibrated per-device. When ground-truth labels are present in the dataset, they take precedence over derived labels.

---

## 4. Train / Validation / Test Split

Splits are **time-based** (no shuffling) to prevent data leakage:

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 70% | Model parameter optimization |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final unbiased performance estimate |

Using chronological splits ensures the model is evaluated on data that occurs *after* the training period — mirroring real deployment conditions.

---

## 5. Model Architecture

```
Input: (batch, seq_len=120, input_size=7)
  │
  ▼
LSTM (input_size=7 → hidden_size=128, num_layers=2, dropout=0.3, bidirectional=False)
  │
  ▼
Last hidden state h_T: (batch, 128)
  │
  ▼
Linear (128 → 4)
  │
  ▼
Logits: (batch, 4)   [softmax applied at inference]
```

**Design choices:**
- **2 LSTM layers**: First layer captures low-level temporal patterns; second layer captures higher-order dynamics.  
- **hidden_size=128**: Balances expressiveness and on-device inference speed.  
- **dropout=0.3**: Applied between LSTM layers (not on the last layer output) to regularize.  
- **Unidirectional**: Real-time inference requires causal processing — no future context.  
- **Last hidden state**: Encodes the full sequence history at the final timestep — appropriate for sequence-to-label tasks.

---

## 6. Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-3 | Good default for AdamW |
| Batch size | 64 | Fits comfortably in memory; stable gradient estimates |
| Max epochs | 50 | Upper bound; early stopping typically triggers earlier |
| Early stopping patience | 5 | Stops after 5 epochs of no val-loss improvement |
| Optimizer | AdamW | Weight decay helps generalization |
| Scheduler | CosineAnnealingLR | Smooth LR decay, avoids oscillation near convergence |
| Seed | 42 | Reproducibility |
| Dropout | 0.3 | Between LSTM layers |
| Class weights | Optional | Inverse-frequency weighting to handle class imbalance |

---

## 7. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall fraction of correct predictions |
| **Macro F1** | Unweighted mean F1 across 4 classes — important for imbalanced datasets |
| **Per-class F1** | F1 for each scenario independently |
| **Confusion Matrix** | 4×4 matrix showing prediction vs. ground truth |
| **Brier Score** | Mean squared error of predicted probabilities vs. one-hot truth — measures calibration |

**Brier Score formula:**
```
BS = (1/N) * Σ Σ (p_kc - y_kc)²
```
where `p_kc` is predicted probability and `y_kc` is 1-hot ground truth.

---

## 8. Export Strategy

### TorchScript Export

```
lstm_forecaster.pt      — TorchScript traced model
feature_meta.json       — Metadata for inference pipeline
```

**`feature_meta.json` schema:**
```json
{
  "feature_cols": [...],
  "window_size": 120,
  "horizon": 60,
  "num_classes": 4,
  "scenario_names": ["Idle", "Power-cycling", "Typing/low-power", "Stress/high-power"],
  "normalizer_params": {
    "cpu_utilization": {"mean": ..., "std": ...},
    ...
  }
}
```

### ONNX Export

```
lstm_forecaster.onnx    — ONNX model for cross-runtime inference
feature_meta.json       — Same metadata file
```

ONNX export enables deployment on non-PyTorch runtimes (ONNX Runtime, TensorRT, CoreML via conversion). Export uses `opset_version=17` with dynamic batch axis.

### Checkpoint Format

During training, checkpoints are saved as:
```python
{
    "state_dict": model.state_dict(),
    "config": config_dict,
    "normalizer": normalizer.to_dict(),  # {"mean": {...}, "std": {...}}
    "feature_cols": FEATURE_COLS
}
```
