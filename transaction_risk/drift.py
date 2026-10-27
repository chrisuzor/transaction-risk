"""Population stability of synthetic windows against the training split.

The reference window is the training split of the default generator. The
unshifted holdout is the test split of that same draw. The shifted window is
a second fixed seed with different feature distributions. Every row is synthetic.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transaction_risk.constants import (
    CHANNEL,
    DRIFT_N_BINS,
    DRIFT_PSI_THRESHOLD,
    DRIFT_RULE,
    DRIFT_SHIFT_SEED,
    MODEL_FEATURES,
    MODEL_VERSION,
    N_ROWS,
    SEED,
    TIMEZONE,
)
from transaction_risk.features import to_model_frame
from transaction_risk.generate import generate_dataset, generate_shifted_dataset
from transaction_risk.scoring import load_artifact, positive_scores

_EPS = 1e-6


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = DRIFT_N_BINS,
) -> float:
    """PSI of `current` against `reference`.

    Bins come from the reference only. Features with at most `n_bins` distinct
    reference values use one bin per value. Other features use quantile bins.
    A zero bin share is floored at 1e-6 and the shares are renormalized.
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI windows must be non-empty")
    edges = _bin_edges(reference, n_bins)
    expected = _shares(reference, edges)
    actual = _shares(current, edges)
    expected = np.maximum(expected, _EPS)
    actual = np.maximum(actual, _EPS)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def is_stable(feature_psi: dict[str, float], score_psi: float) -> bool:
    """True only when every feature PSI and the score PSI are below the threshold."""
    if any(value >= DRIFT_PSI_THRESHOLD for value in feature_psi.values()):
        return False
    return score_psi < DRIFT_PSI_THRESHOLD


def training_reference_and_holdout(
    n_rows: int = N_ROWS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training split and test split from the same draw as model training."""
    frame, labels = generate_dataset(n_rows, seed)
    raw_train, raw_temp, y_train, y_temp = train_test_split(
        frame,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    _raw_val, raw_test, _y_val, _y_test = train_test_split(
        raw_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )
    return raw_train, raw_test


def compare_window(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    artifact: dict,
) -> dict:
    ref_model = to_model_frame(reference)
    cur_model = to_model_frame(current)
    feature_psi = {
        name: population_stability_index(
            ref_model[name].to_numpy(dtype=np.float64),
            cur_model[name].to_numpy(dtype=np.float64),
        )
        for name in MODEL_FEATURES
    }
    score_psi = population_stability_index(
        positive_scores(artifact, reference),
        positive_scores(artifact, current),
    )
    return {
        "feature_psi": feature_psi,
        "score_psi": score_psi,
        "stable": is_stable(feature_psi, score_psi),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
    }


def main() -> None:
    artifact = load_artifact()
    reference, unshifted = training_reference_and_holdout()
    shifted = generate_shifted_dataset(len(unshifted), DRIFT_SHIFT_SEED)
    unshifted_report = compare_window(reference, unshifted, artifact)
    shifted_report = compare_window(reference, shifted, artifact)
    checked_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")
    lines = [
        "drift check",
        f"checked_at={checked_at}",
        f"timezone={TIMEZONE}",
        "data=synthetic",
        f"channel={CHANNEL}",
        f"model_version={MODEL_VERSION}",
        f"reference=training split of generate_dataset(n={N_ROWS}, seed={SEED})",
        f"n_reference={len(reference)}",
        "unshifted=test split of the same draw",
        f"n_unshifted={len(unshifted)}",
        (
            "shifted=generate_shifted_dataset"
            f"(n={len(shifted)}, seed={DRIFT_SHIFT_SEED})"
        ),
        f"n_shifted={len(shifted)}",
        "statistic=population_stability_index",
        f"n_bins={DRIFT_N_BINS}",
        f"psi_threshold={DRIFT_PSI_THRESHOLD:.2f}",
        f"psi_rule={DRIFT_RULE}",
        *_window_lines("unshifted", unshifted_report),
        *_window_lines("shifted", shifted_report),
        f"unshifted_stable={str(unshifted_report['stable']).lower()}",
        f"shifted_stable={str(shifted_report['stable']).lower()}",
    ]
    print("\n".join(lines))
    if unshifted_report["stable"] is not True:
        raise SystemExit("unshifted holdout was treated as drifted")
    if shifted_report["stable"] is not False:
        raise SystemExit("shifted set was treated as stable")


def _window_lines(name: str, report: dict) -> list[str]:
    lines = [
        f"window={name}",
        f"stable={str(report['stable']).lower()}",
        f"score_psi={report['score_psi']:.4f}",
    ]
    for feature, value in report["feature_psi"].items():
        lines.append(f"feature={feature} psi={value:.4f}")
    return lines


def _bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    unique = np.unique(reference)
    if unique.size == 1:
        return np.array([-np.inf, np.inf])
    if unique.size <= n_bins:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
        return np.concatenate(([-np.inf], midpoints, [np.inf]))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles)).astype(np.float64)
    if edges.size < 2:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = float(counts.sum())
    if total == 0.0:
        raise ValueError("histogram produced no counts")
    return counts.astype(np.float64) / total


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
