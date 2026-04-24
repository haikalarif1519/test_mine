from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curve(history, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss_curve.png")
    plt.close()


def plot_confusion_matrix(cm, labels: list[str], step_name: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {step_name}")
    plt.tight_layout()
    plt.savefig(out / f"confusion_matrix_{step_name}.png")
    plt.close()


def plot_class_distribution(labels, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    labels.value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png")
    plt.close()
