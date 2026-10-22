"""Configuration objects describing one experiment."""
from dataclasses import dataclass, field, asdict

DEFAULT_PAST_COLUMNS = (
    "power", "cadence", "heart_rate",
    "TT_10", "PP_10", "RF_10",
    "seconds_since_session_start", "had_warmup",
    "day_of_week_sin", "day_of_week_cos",
    "time_of_day_sin", "time_of_day_cos",
)
DEFAULT_FUTURE_COLUMNS = (
    "power", "cadence",
    "seconds_since_session_start",
    "time_of_day_sin", "time_of_day_cos",
)


@dataclass(frozen=True)
class DataConfig:
    """Settings that change the windows fed into the model."""
    past_steps: int = 120          # 10 min at 5 s
    future_steps: int = 12         # 1 min at 5 s
    train_stride: int = 2
    evaluation_stride: int = 12
    past_columns: tuple[str, ...] = DEFAULT_PAST_COLUMNS
    future_columns: tuple[str, ...] = DEFAULT_FUTURE_COLUMNS
    target_column: str = "heart_rate"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters."""
    gru_units: int = 64
    dense_units: int = 64
    dropout_rate: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and training loop settings."""
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete description of one experiment."""
    name: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return asdict(self)