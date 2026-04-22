from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_lstm_model(config: dict) -> Model:
    mdl_cfg = config["model"]
    lookback = config["preprocessing"]["lookback"]
    feature_count = len(config["feature_columns"])

    inputs = Input(shape=(lookback, feature_count))
    x = LSTM(mdl_cfg["lstm_units_layer1"], return_sequences=True)(inputs)
    x = LSTM(mdl_cfg["lstm_units_layer2"], return_sequences=False)(x)
    x = Dropout(mdl_cfg["dropout"])(x)
    x = RepeatVector(mdl_cfg["output_steps"])(x)
    x = LSTM(mdl_cfg["lstm_units_layer2"], return_sequences=True)(x)
    outputs = TimeDistributed(Dense(mdl_cfg["output_classes"], activation="softmax"))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=mdl_cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
