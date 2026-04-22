from tensorflow.keras.models import load_model as keras_load_model


CLASS_NAMES = ["Idle", "Low Load", "High Load"]
STEP_NAMES = ["t+05", "t+10", "t+15"]


def load_model(path: str):
    return keras_load_model(path)


def predict_trajectory(model, x_input):
    pred = model.predict(x_input, verbose=0)
    output = {}
    for i, step in enumerate(STEP_NAMES):
        probs = pred[0, i]
        output[step] = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}
    return output
