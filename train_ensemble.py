"""Command-line entry point for training one dashcam ensemble member."""

import argparse

from dashcam_ensemble.config import MODEL_CONFIGS
from dashcam_ensemble.training import train_ensemble_member


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one VideoMAE dashcam classification model."
    )
    parser.add_argument(
        "seed",
        type=int,
        choices=sorted(MODEL_CONFIGS),
        help="Ensemble member configuration to train.",
    )
    parser.add_argument("--train-csv", required=True, help="Path to the training CSV.")
    parser.add_argument("--video-dir", required=True, help="Directory containing MP4 videos.")
    parser.add_argument(
        "--clips-dir",
        required=True,
        help="Directory containing per-clip JSON metadata.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (defaults to ./ensemble_temp_seed_all_<seed>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = train_ensemble_member(
        seed=args.seed,
        train_csv=args.train_csv,
        video_dir=args.video_dir,
        clips_dir=args.clips_dir,
        output_dir=args.output_dir,
    )
    print(f"Training complete. Model saved to: {model_dir}")


if __name__ == "__main__":
    main()
