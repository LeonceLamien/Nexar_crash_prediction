"""Training pipeline for a single member of the ensemble."""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments

from dashcam_ensemble.config import MODEL_CONFIGS, get_model_config
from dashcam_ensemble.data import LiveDashcamDataset
from dashcam_ensemble.model import VideoMAEClassifier


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_ensemble_member(
    seed: int,
    train_csv: str,
    video_dir: str,
    clips_dir: str,
    output_dir: str | None = None,
) -> str:
    """Train and save one configured ensemble member."""
    hyperparameters = get_model_config(seed)
    set_seed(seed)

    output_dir = output_dir or f"./ensemble_temp_seed_all_{seed}"
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(train_csv).reset_index(drop=True)
    if "id" not in train_df.columns:
        raise ValueError(f"The training CSV must contain an 'id' column: {train_csv}")

    train_dataset = LiveDashcamDataset(
        df=train_df,
        video_dir=video_dir,
        clips_dir=clips_dir,
        mode="train",
        num_frames=16,
        full_augmentation=bool(hyperparameters["full_augmentation"]),
    )
    model = VideoMAEClassifier(
        num_labels=2,
        dropout=float(hyperparameters["dropout"]),
        label_smoothing=float(hyperparameters["label_smoothing"]),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(hyperparameters["epochs"]),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=6,
        learning_rate=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
        warmup_ratio=float(hyperparameters["warmup_ratio"]),
        lr_scheduler_type="cosine",
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        logging_strategy="steps",
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)

    metrics = {
        "seed": seed,
        "model_index": list(MODEL_CONFIGS).index(seed),
        "trained_on": "full training dataset",
        "hyperparameters": hyperparameters,
    }
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metrics, metadata_file, indent=2)

    return final_model_dir
