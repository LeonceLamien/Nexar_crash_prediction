"""Video dataset and data augmentation utilities."""

import json
import os
import random

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def temporal_augmentation(
    frame_indices: list[int],
    total_frames: int | None = None,
    max_jitter: int = 3,
) -> list[int]:
    """Apply bounded random temporal jitter to frame indices."""
    augmented_indices = []
    for index in frame_indices:
        new_index = index + random.randint(-max_jitter, max_jitter)
        if total_frames is not None:
            new_index = max(0, min(new_index, total_frames - 1))
        else:
            new_index = max(0, new_index)
        augmented_indices.append(new_index)
    return augmented_indices


class LiveDashcamDataset(Dataset):
    """Load three annotated video clips per video for binary classification."""

    def __init__(
        self,
        df,
        video_dir: str,
        clips_dir: str,
        mode: str = "train",
        num_frames: int = 16,
        full_augmentation: bool = False,
    ):
        if mode not in {"train", "val"}:
            raise ValueError("mode must be either 'train' or 'val'.")

        self.df = df
        self.video_dir = video_dir
        self.clips_dir = clips_dir
        self.mode = mode
        self.num_frames = num_frames
        self.clip_cache: dict[str, dict] = {}
        self._preload_clip_indices()

        aug_layers = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
        ]
        if mode == "train" and full_augmentation:
            aug_layers.extend(
                [
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        aug_layers.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.transform = transforms.Compose(aug_layers)

    def _preload_clip_indices(self) -> None:
        video_ids = set(self.df["id"].astype(str).str.zfill(5).values)
        for filename in os.listdir(self.clips_dir):
            if not filename.endswith(".json"):
                continue

            parts = filename[:-5].split("_")
            video_id = parts[0]
            if video_id not in video_ids:
                continue
            if len(parts) < 3:
                raise ValueError(
                    f"Clip metadata filename must include a clip index: {filename}"
                )

            clip_index = parts[2]
            path = os.path.join(self.clips_dir, filename)
            with open(path, encoding="utf-8") as metadata_file:
                clip_data = json.load(metadata_file)
            self.clip_cache[f"{video_id}_{clip_index}"] = clip_data

    def __len__(self) -> int:
        return len(self.df) * 3

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        video_index = index // 3
        clip_index = index % 3
        video_id = self.df["id"].iloc[video_index]
        video_id_str = str(int(video_id)).zfill(5)
        key = f"{video_id_str}_{clip_index}"

        if key not in self.clip_cache:
            raise KeyError(
                f"Missing clip metadata for video {video_id_str}, "
                f"clip {clip_index} in {self.clips_dir}."
            )

        clip_data = self.clip_cache[key]
        frame_indices = clip_data["indices"]
        label = clip_data["videomae_label"]
        video_path = os.path.join(self.video_dir, f"{video_id_str}.mp4")

        if self.mode == "train":
            frame_count_capture = cv2.VideoCapture(video_path)
            if not frame_count_capture.isOpened():
                frame_count_capture.release()
                raise OSError(f"Could not open video file: {video_path}")
            total_frames = int(
                frame_count_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            frame_count_capture.release()
            frame_indices = temporal_augmentation(
                frame_indices, total_frames, max_jitter=3
            )

        sorted_pairs = sorted(enumerate(frame_indices), key=lambda pair: pair[1])
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Could not open video file: {video_path}")

        frames_by_position = {}
        last_frame_index = -1
        for original_position, frame_index in sorted_pairs:
            if frame_index > last_frame_index and frame_index - last_frame_index < 5:
                for _ in range(frame_index - last_frame_index):
                    success, frame = capture.read()
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()

            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_by_position[original_position] = self.transform(frame_rgb)
            else:
                frames_by_position[original_position] = None
            last_frame_index = frame_index
        capture.release()

        frames = []
        for position in range(len(frame_indices)):
            frame = frames_by_position[position]
            if frame is not None:
                frames.append(frame)
            else:
                frames.append(
                    frames[-1] if frames else torch.zeros(3, 224, 224)
                )

        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))

        return {
            "pixel_values": torch.stack(frames[: self.num_frames]),
            "labels": torch.tensor(label, dtype=torch.long),
        }
