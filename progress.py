"""Progress display for experiment runs."""
import math

import keras
from tqdm.auto import tqdm


class EpochProgressBar(keras.callbacks.Callback):
    """Show one compact progress bar per training run instead of Keras' log lines."""

    def __init__(self, total_epochs: int, description: str = "epochs"):
        super().__init__()
        self.total_epochs = total_epochs
        self.description = description
        self.progress_bar = None
        self.best_validation_loss = math.inf

    def on_train_begin(self, logs=None):
        self.best_validation_loss = math.inf
        self.progress_bar = tqdm(
            total=self.total_epochs,
            desc=f"  {self.description}",
            unit="epoch",
            position=1,
            leave=False,  # bar disappears after the run, outer bar stays
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        validation_loss = logs.get("val_loss", math.nan)
        self.best_validation_loss = min(self.best_validation_loss, validation_loss)
        learning_rate = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))

        self.progress_bar.set_postfix(
            loss=f"{logs.get('loss', math.nan):.3f}",
            val_loss=f"{validation_loss:.3f}",
            best=f"{self.best_validation_loss:.3f}",
            lr=f"{learning_rate:.1e}",
        )
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        # Early stopping ends before max_epochs, so the bar is closed explicitly
        self.progress_bar.close()
