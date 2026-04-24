import torch
import torch.nn.functional as F


CLASS_NAMES = ["Idle", "Low Load", "High Load", "Power Cycle"]
STEP_NAMES = ["t+05", "t+10", "t+15"]


def load_model(path: str):
    model = torch.load(path, weights_only=False)
    model.eval()
    return model


def predict_trajectory(model, x_input):
    x_tensor = torch.tensor(x_input, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)                              # (1, steps, classes)
        probs = F.softmax(logits, dim=-1).numpy()             # (1, steps, classes)
    output = {}
    for i, step in enumerate(STEP_NAMES):
        output[step] = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs[0, i])}
    return output
