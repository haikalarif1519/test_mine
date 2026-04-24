import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, X_train, y_train, X_val, y_val, config: dict) -> dict:
    """Train the model and return a history dict with 'loss', 'val_loss', and 'val_macro_f1' lists.

    Class-imbalance handling
    ------------------------
    Inverse-frequency weights are computed from the training labels (using the
    first prediction step as a representative distribution) and passed to
    CrossEntropyLoss so that minority classes (Low Load, High Load) contribute
    proportionally more gradient signal than the dominant Idle class.

    Early stopping is driven by macro-averaged F1 on the validation set
    (higher is better) rather than val_loss, because val_loss rewards a model
    that simply predicts Idle for every sample.
    """
    paths = config["paths"]
    mdl_cfg = config["model"]

    Path(paths["saved_models"]).mkdir(parents=True, exist_ok=True)
    Path(paths["logs"]).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=mdl_cfg["batch_size"],
        shuffle=False,  # Preserve temporal order; sequences must not be shuffled.
    )

    optimizer = Adam(model.parameters(), lr=mdl_cfg["learning_rate"])

    # Compute inverse-frequency class weights from the training distribution
    # (first predict step used as a representative proxy for the label mix).
    num_classes = mdl_cfg["output_classes"]
    counts = np.bincount(y_train[:, 0], minlength=num_classes).astype(float)
    weights = 1.0 / counts.clip(min=1)
    weights /= weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    patience_counter = 0
    best_state = None
    history = {"loss": [], "val_loss": [], "val_macro_f1": []}

    log_path = Path(paths["logs"]) / "training_metrics.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_loss", "val_macro_f1"])

    for epoch in range(mdl_cfg["epochs"]):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)                           # (N, steps, classes)
            loss = criterion(logits.permute(0, 2, 1), y_batch)  # (N, classes, steps)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        epoch_loss /= len(X_train_t)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = criterion(val_logits.permute(0, 2, 1), y_val_t.to(device)).item()
            val_pred = val_logits.argmax(dim=-1).cpu().numpy().reshape(-1)
            val_true = y_val_t.numpy().reshape(-1)

        val_macro_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))

        history["loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        print(
            f"Epoch {epoch + 1}/{mdl_cfg['epochs']} - "
            f"loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_macro_f1: {val_macro_f1:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, epoch_loss, val_loss, val_macro_f1])

        # --- Checkpoint: save on improvement of macro-F1 (not val_loss) ---
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.cpu(), paths["model"])
            model = model.to(device)
        else:
            patience_counter += 1
            if patience_counter >= mdl_cfg["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()

    return history
