"""Recall-oriented threshold on a fixed score grid."""

import numpy as np

from transaction_risk.constants import TARGET_RECALL


def choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> float:
    """Highest 0.001 grid point whose recall is at least the target.

    Search runs from high scores toward low scores and stops at the first
    point that meets the recall target. That is the strictest cutoff that
    still keeps the requested recall on this split.
    """
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("cannot choose a threshold with no positive labels")

    grid = np.round(np.arange(0.001, 1.0, 0.001), 3)
    for threshold in grid[::-1]:
        true_positives = int(np.sum((scores >= threshold) & (labels == 1)))
        recall = true_positives / positives
        if recall >= target_recall:
            return float(f"{float(threshold):.3f}")

    raise ValueError(
        f"no threshold on the 0.001 grid reaches recall {target_recall:.2f}"
    )
