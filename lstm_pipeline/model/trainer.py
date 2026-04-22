from pathlib import Path

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint


def train_model(model, X_train, y_train, X_val, y_val, config: dict):
    paths = config["paths"]
    mdl_cfg = config["model"]

    Path(paths["saved_models"]).mkdir(parents=True, exist_ok=True)
    Path(paths["logs"]).mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=mdl_cfg["patience"], restore_best_weights=True),
        ModelCheckpoint(paths["model"], monitor="val_loss", save_best_only=True),
        CSVLogger(str(Path(paths["logs"]) / "training_metrics.csv")),
    ]

    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=mdl_cfg["epochs"],
        batch_size=mdl_cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )
