"""Evaluation and probability-calibration metrics."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute average precision and ROC AUC from Trainer predictions."""
    logits, labels = eval_pred
    probabilities = torch.softmax(torch.as_tensor(logits), dim=-1)[:, 1].numpy()
    average_precision = float(average_precision_score(labels, probabilities))
    auc_roc = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) > 1
        else 0.5
    )
    return {"average_precision": average_precision, "auc_roc": auc_roc}


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error for the predicted class."""
    with torch.no_grad():
        probabilities = torch.softmax(logits / temperature, dim=-1)
        confidence, predictions = probabilities.max(dim=-1)
        accuracy = predictions.eq(labels).float()
        ece = torch.zeros(1)
        bins = torch.linspace(0, 1, n_bins + 1)
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (confidence > lower) & (confidence <= upper)
            if mask.any():
                ece += (
                    (accuracy[mask].mean() - confidence[mask].mean()).abs()
                    * mask.float().mean()
                )
    return ece.item()


def learn_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Learn a positive scalar temperature by minimizing validation NLL."""
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=100)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.exp(log_temperature).item()
