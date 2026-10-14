"""VideoMAE model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification


class VideoMAEClassifier(nn.Module):
    """VideoMAE video classifier with optional label-smoothed loss."""

    def __init__(
        self,
        num_labels: int = 2,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.videomae = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-large-finetuned-kinetics",
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        hidden_size = self.videomae.config.hidden_size
        self.videomae.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.label_smoothing = label_smoothing
        self.num_labels = num_labels

    def smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels, self.num_labels).float()
        return (
            one_hot * (1 - self.label_smoothing)
            + self.label_smoothing / self.num_labels
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.videomae(pixel_values=pixel_values)
        logits = outputs.logits

        loss = None
        if labels is not None:
            if self.training and self.label_smoothing > 0:
                smoothed_labels = self.smooth_labels(labels)
                loss = -(
                    smoothed_labels * F.log_softmax(logits, dim=-1)
                ).sum(dim=-1).mean()
            else:
                loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
