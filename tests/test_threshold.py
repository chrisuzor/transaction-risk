import numpy as np
import pytest

from transaction_risk.threshold import choose_threshold


def test_choose_threshold_is_the_highest_recall_qualified_grid_point():
    labels = np.array([0, 0, 1, 1, 1, 1])
    scores = np.array([0.05, 0.40, 0.20, 0.50, 0.70, 0.95])
    assert choose_threshold(labels, scores, target_recall=0.75) == 0.5
    assert choose_threshold(labels, scores, target_recall=1.0) == 0.2


def test_choose_threshold_rejects_empty_positive_class():
    with pytest.raises(ValueError):
        choose_threshold(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_choose_threshold_raises_when_target_is_impossible():
    with pytest.raises(ValueError):
        choose_threshold(np.array([1, 0]), np.array([0.0, 0.2]), target_recall=0.90)
