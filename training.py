"""Train, evaluate and persist a single run."""
import json
from datetime import datetime
from pathlib import Path

import joblib
import keras
import numpy as np

from config import ExperimentConfig
from data_preparation import WindowedData, SCALED_COLUMNS
from model_factory import build_forecast_model
from utils import persistence_forecast, mae_in_bpm, column_scale

from progress import EpochProgressBar


def create_callbacks(config: ExperimentConfig, run_directory: Path, seed: int) -> list:
    training = config.training
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=training.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=training.reduce_lr_factor,
            patience=training.reduce_lr_patience,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=run_directory / "best_model.keras",
            monitor="val_loss", save_best_only=True,
        ),
        keras.callbacks.CSVLogger(run_directory / "training_log.csv"),
        EpochProgressBar(
            total_epochs=config.training.max_epochs,
            description=f"{config.name} seed {seed}",
        ),
    ]


def evaluate(model: keras.Model, past: np.ndarray, future: np.ndarray, target: np.ndarray,
             heart_rate_index: int, bpm_scale: float) -> dict:
    """Return model and persistence MAE in bpm."""
    prediction = model.predict({"past": past, "future": future}, verbose=0)
    model_errors = mae_in_bpm(prediction, target, bpm_scale)
    baseline_errors = mae_in_bpm(
        persistence_forecast(past, heart_rate_index, target.shape[1]), target, bpm_scale
    )
    return {
        "model_mae_bpm": float(model_errors.mean()),
        "model_mae_60s_bpm": float(model_errors[-1]),
        "baseline_mae_bpm": float(baseline_errors.mean()),
    }


def run_experiment(config: ExperimentConfig, data: WindowedData, scaler,
                   seed: int, output_directory: Path = Path("models")) -> dict:
    """Train one configuration with one seed and return a flat result record."""
    keras.utils.set_random_seed(seed)
    run_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{config.name}_seed{seed}"
    run_directory = output_directory / run_name
    run_directory.mkdir(parents=True, exist_ok=True)

    model = build_forecast_model(
        config.model,
        past_shape=data.past_train.shape[1:],
        future_shape=data.future_train.shape[1:],
        heart_rate_index=data.heart_rate_index,
    )
    model.compile(optimizer=keras.optimizers.Adam(config.training.learning_rate), loss="mae")

    history = model.fit(
        x={"past": data.past_train, "future": data.future_train},
        y=data.target_train,
        validation_data=({"past": data.past_validation, "future": data.future_validation},
                         data.target_validation),
        epochs=config.training.max_epochs,
        batch_size=config.training.batch_size,
        callbacks=create_callbacks(config, run_directory, seed),
        verbose=0,
    )

    bpm_scale = column_scale(scaler, SCALED_COLUMNS, "heart_rate")
    best_epoch = int(np.argmin(history.history["val_loss"]))
    validation_metrics = evaluate(model, data.past_validation, data.future_validation,
                                  data.target_validation, data.heart_rate_index, bpm_scale)

    save_run_artifacts(run_directory, config, data, scaler, seed, best_epoch, validation_metrics)

    return {
        "experiment": config.name,
        "seed": seed,
        "run_directory": str(run_directory),
        "best_epoch": best_epoch + 1,
        "train_loss_at_best": history.history["loss"][best_epoch],
        "val_loss_best": history.history["val_loss"][best_epoch],
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
    }


def save_run_artifacts(run_directory: Path, config: ExperimentConfig, data: WindowedData,
                       scaler, seed: int, best_epoch: int, validation_metrics: dict) -> None:
    """Store everything needed to reload and evaluate the model later."""
    joblib.dump(scaler, run_directory / "scaler.joblib")
    run_config = {
        **config.to_dict(),
        "seed": seed,
        "scaled_columns": SCALED_COLUMNS,
        "heart_rate_index": data.heart_rate_index,
        "best_epoch": best_epoch + 1,
        "validation": validation_metrics,
    }
    with open(run_directory / "config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    np.savez_compressed(run_directory / "test_data.npz",
                        past=data.past_test, future=data.future_test, target=data.target_test)