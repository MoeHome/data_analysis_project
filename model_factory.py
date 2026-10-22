"""Model architectures."""
import keras
from keras import layers

from config import ModelConfig


def build_forecast_model(
        model_config: ModelConfig,
        past_shape: tuple[int, int],
        future_shape: tuple[int, int],
        heart_rate_index: int,
) -> keras.Model:
    """GRU encoder over the past plus flattened future inputs, residual output."""
    future_steps = future_shape[0]
    past_input = keras.Input(shape=past_shape, name="past")
    future_input = keras.Input(shape=future_shape, name="future")

    encoded_past = layers.GRU(model_config.gru_units, name="past_encoder")(past_input)
    flat_future = layers.Flatten(name="future_flat")(future_input)

    combined = layers.Concatenate(name="combined")([encoded_past, flat_future])
    hidden = layers.Dense(model_config.dense_units, activation="relu")(combined)
    hidden = layers.Dropout(model_config.dropout_rate)(hidden)
    heart_rate_delta = layers.Dense(future_steps, name="heart_rate_delta")(hidden)

    last_heart_rate = past_input[:, -1, heart_rate_index: heart_rate_index + 1]
    forecast = layers.Add(name="heart_rate_forecast")([heart_rate_delta, last_heart_rate])

    return keras.Model(inputs=[past_input, future_input], outputs=forecast)
