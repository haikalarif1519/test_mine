import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix


CLASS_LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["Idle", "Low Load", "High Load", "Power Cycle"]


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        y_prob = F.softmax(logits, dim=-1).numpy()

    y_pred = np.argmax(y_prob, axis=-1)

    step_names = ["t+05", "t+10", "t+15"]

    results = {}
    for i, step in enumerate(step_names):
        true_i = y_test[:, i]
        pred_i = y_pred[:, i]
        cm = confusion_matrix(true_i, pred_i, labels=CLASS_LABELS)
        report = classification_report(
            true_i, pred_i, labels=CLASS_LABELS, target_names=CLASS_NAMES,
            zero_division=0, output_dict=True,
        )

        high_mask = true_i == 2
        high_mis_rate = float(((pred_i[high_mask] == 0).mean() if high_mask.any() else 0.0) * 100)
        flagged = high_mis_rate > 2.0

        pc_mask = true_i == 3
        pc_mis_rate = float(((pred_i[pc_mask] == 0).mean() if pc_mask.any() else 0.0) * 100)
        pc_flagged = pc_mis_rate > 5.0

        macro_f1 = report["macro avg"]["f1-score"]

        print(f"\n=== {step} ===")
        print("Confusion Matrix:\n", cm)
        print(f"Macro-averaged F1: {macro_f1:.4f}")
        print(f"High Load -> Idle rate: {high_mis_rate:.2f}% {'[FLAGGED]' if flagged else '[OK]'}")
        print(f"Power Cycle -> Idle rate: {pc_mis_rate:.2f}% {'[FLAGGED]' if pc_flagged else '[OK]'}")
        print("\nPer-class metrics:")
        for name in CLASS_NAMES:
            r = report.get(name, {})
            print(
                f"  {name}: precision={r.get('precision', 0):.3f}  "
                f"recall={r.get('recall', 0):.3f}  f1={r.get('f1-score', 0):.3f}  "
                f"support={int(r.get('support', 0))}"
            )

        results[step] = {
            "confusion_matrix": cm,
            "classification_report": report,
            "macro_f1": macro_f1,
            "high_load_as_idle_rate_pct": high_mis_rate,
            "flagged": flagged,
            "power_cycle_as_idle_rate_pct": pc_mis_rate,
            "power_cycle_flagged": pc_flagged,
        }
    return results
