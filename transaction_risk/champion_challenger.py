"""Offline champion/challenger comparison on one fixed synthetic batch.

The champion is baseline-v1. Its threshold decision is the one that would be
served. The challenger is the same scores with a shadow threshold of 0.50.
A gradient boosting model was a worse fit on this seed, so the challenger is
not a second algorithm. Every row is synthetic. Nothing here is live traffic.
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
    CHALLENGER_THRESHOLD,
    CHANNEL,
    MODEL_VERSION,
    N_ROWS,
    SEED,
    TIMEZONE,
)
from transaction_risk.generate import generate_dataset
from transaction_risk.scoring import load_artifact, positive_scores


def fixed_test_batch(
    n_rows: int = N_ROWS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Test split of the same draw and split as training."""
    frame, labels = generate_dataset(n_rows, seed)
    _raw_train, raw_temp, _y_train, y_temp = train_test_split(
        frame,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    _raw_val, raw_test, _y_val, y_test = train_test_split(
        raw_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )
    return raw_test, np.asarray(y_test).astype(int)


def shadow_compare(
    frame: pd.DataFrame,
    labels: np.ndarray,
    artifact: dict,
    challenger_threshold: float = CHALLENGER_THRESHOLD,
) -> dict:
    """Score one batch. Record challenger fields without serving them."""
    scores = positive_scores(artifact, frame)
    labels = np.asarray(labels).astype(int)
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    champion_threshold = float(artifact["threshold"])
    records = []
    for score, label in zip(scores, labels, strict=True):
        value = float(score)
        champion_decision = "flag" if value >= champion_threshold else "allow"
        challenger_decision = "flag" if value >= challenger_threshold else "allow"
        records.append(
            {
                "champion_score": value,
                "champion_decision": champion_decision,
                "challenger_score": value,
                "challenger_decision": challenger_decision,
                "disagree": champion_decision != challenger_decision,
                "served_decision": champion_decision,
                "label": int(label),
            }
        )
    champion_fp, champion_fn = _false_counts(
        labels, scores, champion_threshold
    )
    challenger_fp, challenger_fn = _false_counts(
        labels, scores, challenger_threshold
    )
    disagreement_count = sum(1 for row in records if row["disagree"])
    agreement_count = len(records) - disagreement_count
    return {
        "records": records,
        "n_batch": len(records),
        "champion_threshold": champion_threshold,
        "challenger_threshold": float(challenger_threshold),
        "agreement_count": agreement_count,
        "agreement_rate": agreement_count / len(records),
        "disagreement_count": disagreement_count,
        "champion_false_positives": champion_fp,
        "champion_false_negatives": champion_fn,
        "challenger_false_positives": challenger_fp,
        "challenger_false_negatives": challenger_fn,
    }


def compare_fixed_batch(artifact: dict | None = None) -> dict:
    frame, labels = fixed_test_batch()
    return shadow_compare(frame, labels, artifact or load_artifact())


def main() -> None:
    report = compare_fixed_batch()
    checked_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")
    lines = [
        "champion_challenger",
        f"checked_at={checked_at}",
        f"timezone={TIMEZONE}",
        "data=synthetic",
        f"channel={CHANNEL}",
        f"batch=test split of generate_dataset(n={N_ROWS}, seed={SEED})",
        f"n_batch={report['n_batch']}",
        f"champion_model={MODEL_VERSION}",
        "champion_type=logistic_regression",
        f"champion_threshold={report['champion_threshold']:.3f}",
        "champion_role=served",
        "challenger_type=second_threshold",
        f"challenger_threshold={report['challenger_threshold']:.2f}",
        "challenger_role=shadow_only",
        "shadow_record=champion_score,champion_decision,challenger_score,challenger_decision,disagree",
        f"agreement_count={report['agreement_count']}",
        f"agreement_rate={report['agreement_rate']:.4f}",
        f"disagreement_count={report['disagreement_count']}",
        f"champion_false_positives={report['champion_false_positives']}",
        f"champion_false_negatives={report['champion_false_negatives']}",
        f"challenger_false_positives={report['challenger_false_positives']}",
        f"challenger_false_negatives={report['challenger_false_negatives']}",
    ]
    print("\n".join(lines))


def _false_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[int, int]:
    flagged = np.asarray(scores) >= threshold
    y = np.asarray(labels).astype(bool)
    false_positives = int(np.sum(flagged & ~y))
    false_negatives = int(np.sum(~flagged & y))
    return false_positives, false_negatives


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
