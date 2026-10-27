import numpy as np

from transaction_risk.constants import DRIFT_PSI_THRESHOLD, DRIFT_RULE, DRIFT_SHIFT_SEED, N_ROWS, SEED
from transaction_risk.drift import (
    compare_window,
    is_stable,
    population_stability_index,
    training_reference_and_holdout,
)
from transaction_risk.generate import generate_shifted_dataset
from transaction_risk.scoring import load_artifact


def test_identical_samples_have_no_population_shift():
    values = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 0.3, 0.7, 0.1, 0.9, 0.4])
    assert population_stability_index(values, values.copy()) == 0.0


def test_population_stability_index_flags_a_large_shift():
    reference = np.array([0.0, 1.0] * 500)
    current = np.ones(1000)
    assert population_stability_index(reference, current) >= DRIFT_PSI_THRESHOLD


def test_stability_rule_uses_the_documented_threshold():
    assert is_stable({"log_amount": DRIFT_PSI_THRESHOLD - 0.01}, score_psi=0.0) is True
    assert is_stable({"log_amount": DRIFT_PSI_THRESHOLD}, score_psi=0.0) is False
    assert is_stable({"log_amount": 0.0}, score_psi=DRIFT_PSI_THRESHOLD) is False
    assert DRIFT_RULE.endswith(f"{DRIFT_PSI_THRESHOLD:.2f}")


def test_unshifted_holdout_is_stable():
    reference, holdout = training_reference_and_holdout()
    report = compare_window(reference, holdout, load_artifact())
    assert len(reference) == 14_000
    assert len(holdout) == 3_000
    assert report["stable"] is True
    assert report["score_psi"] < DRIFT_PSI_THRESHOLD
    assert all(value < DRIFT_PSI_THRESHOLD for value in report["feature_psi"].values())


def test_shifted_set_is_not_treated_as_stable():
    reference, holdout = training_reference_and_holdout()
    shifted = generate_shifted_dataset(len(holdout), DRIFT_SHIFT_SEED)
    report = compare_window(reference, shifted, load_artifact())
    assert report["stable"] is False
    assert (
        report["score_psi"] >= DRIFT_PSI_THRESHOLD
        or max(report["feature_psi"].values()) >= DRIFT_PSI_THRESHOLD
    )


def test_windows_come_from_the_training_generator():
    reference, holdout = training_reference_and_holdout(N_ROWS, SEED)
    again_reference, again_holdout = training_reference_and_holdout(N_ROWS, SEED)
    assert reference.reset_index(drop=True).equals(again_reference.reset_index(drop=True))
    assert holdout.reset_index(drop=True).equals(again_holdout.reset_index(drop=True))
