"""Training configurations for the ensemble members."""

MODEL_CONFIGS: dict[int, dict[str, float | int | bool]] = {
    42: {
        "learning_rate": 9.03e-06,
        "dropout": 0.1,
        "label_smoothing": 0.30,
        "weight_decay": 2.90e-04,
        "warmup_ratio": 0.10,
        "epochs": 3,
        "full_augmentation": False,
    },
    343: {
        "learning_rate": 1.86e-05,
        "dropout": 0.1,
        "label_smoothing": 0.15,
        "weight_decay": 3.39e-04,
        "warmup_ratio": 0.20,
        "epochs": 3,
        "full_augmentation": True,
    },
    643: {
        "learning_rate": 9.03e-06,
        "dropout": 0.1,
        "label_smoothing": 0.30,
        "weight_decay": 2.90e-04,
        "warmup_ratio": 0.10,
        "epochs": 3,
        "full_augmentation": False,
    },
}


def get_model_config(seed: int) -> dict[str, float | int | bool]:
    """Return the hyperparameters associated with an ensemble seed."""
    try:
        return MODEL_CONFIGS[seed]
    except KeyError as exc:
        valid_seeds = ", ".join(str(value) for value in MODEL_CONFIGS)
        raise ValueError(
            f"Unknown seed {seed}. Choose one of: {valid_seeds}."
        ) from exc
