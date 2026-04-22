import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=-1)

    labels = [0, 1, 2]
    target_names = ["Idle", "Low Load", "High Load"]
    step_names = ["t+05", "t+10", "t+15"]

    results = {}
    for i, step in enumerate(step_names):
        true_i = y_test[:, i]
        pred_i = y_pred[:, i]
        cm = confusion_matrix(true_i, pred_i, labels=labels)
        report = classification_report(true_i, pred_i, labels=labels, target_names=target_names, zero_division=0, output_dict=True)

        high_mask = true_i == 2
        mis_rate = float(((pred_i[high_mask] == 0).mean() if high_mask.any() else 0.0) * 100)
        flagged = mis_rate > 2.0

        print(f"\n=== {step} ===")
        print("Confusion Matrix:\n", cm)
        print(f"High Load -> Idle rate: {mis_rate:.2f}% {'[FLAGGED]' if flagged else '[OK]'}")

        results[step] = {
            "confusion_matrix": cm,
            "classification_report": report,
            "high_load_as_idle_rate_pct": mis_rate,
            "flagged": flagged,
        }
    return results
