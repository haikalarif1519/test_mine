import joblib


def load_scaler(config: dict):
    return joblib.load(config["paths"]["scaler"])


def normalize_inference_features(df, scaler, config: dict):
    out = df.copy()
    cols = config["scale_columns"]
    out[cols] = scaler.transform(out[cols].astype(float))
    return out


def reshape_for_lstm(df):
    arr = df.to_numpy(dtype=float)
    return arr.reshape(1, arr.shape[0], arr.shape[1])
