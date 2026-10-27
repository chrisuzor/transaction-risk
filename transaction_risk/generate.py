"""Fixed-seed synthetic card-not-present transactions.

Every row is synthetic. Values are drawn from the random distributions in
this module. Nothing here is read from a real payment file, card number,
cardholder attribute, scrape, or production metric.
"""

import numpy as np
import pandas as pd

from transaction_risk.constants import (
    KNOWN_EXAMPLE_INDEX,
    LABEL_INTERCEPT,
    LABEL_NOISE,
    LABEL_WEIGHTS,
    MODEL_FEATURES,
    N_ROWS,
    RAW_FIELDS,
    SEED,
)
from transaction_risk.features import to_model_frame


def generate_dataset(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if n < 10:
        raise ValueError("n must be at least 10")
    rng = np.random.default_rng(seed)
    frame = _draw_frame(rng, n)
    labels = _draw_labels(rng, frame)
    return frame, labels


def known_example() -> dict:
    """Row 0 of the default dataset. Stable for the pinned seed and row count."""
    frame, _ = generate_dataset(N_ROWS, SEED)
    return row_to_payload(frame.iloc[KNOWN_EXAMPLE_INDEX])


def payloads_at(indices: list[int]) -> list[dict]:
    frame, _ = generate_dataset(N_ROWS, SEED)
    return [row_to_payload(frame.iloc[i]) for i in indices]


def row_to_payload(row: pd.Series) -> dict:
    return {
        "amount": round(float(row["amount"]), 2),
        "hour_of_day": int(row["hour_of_day"]),
        "day_of_week": int(row["day_of_week"]),
        "merchant_category": int(row["merchant_category"]),
        "international": bool(row["international"]),
        "billing_shipping_mismatch": bool(row["billing_shipping_mismatch"]),
        "txn_count_1h": int(row["txn_count_1h"]),
        "txn_count_24h": int(row["txn_count_24h"]),
        "mean_amount_7d": round(float(row["mean_amount_7d"]), 2),
        "device_age_days": int(row["device_age_days"]),
        "account_age_days": int(row["account_age_days"]),
        "km_from_prev": round(float(row["km_from_prev"]), 2),
    }


def _draw_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # Draw order is part of the seed contract. Inserting a draw changes row 0.
    amount = np.clip(rng.lognormal(mean=3.6, sigma=0.75, size=n), 0.5, 25_000.0)
    hour = rng.integers(0, 24, size=n)
    dow = rng.integers(0, 7, size=n)
    merchant = rng.integers(0, 10, size=n)
    international = rng.random(n) < 0.12
    mismatch = rng.random(n) < 0.07
    txn_1h = np.clip(rng.poisson(0.35, size=n), 0, 30)
    txn_24h = np.clip(txn_1h + rng.poisson(1.4, size=n), 0, 80)
    txn_24h = np.maximum(txn_24h, txn_1h)
    mean_amount = np.clip(rng.lognormal(3.4, 0.45, size=n), 0.5, 10_000.0)
    device_age = rng.integers(0, 1800, size=n)
    account_age = rng.integers(1, 2500, size=n)
    km = rng.exponential(40.0, size=n)
    far = rng.random(n) < 0.06
    bump = rng.uniform(200.0, 5000.0, size=n)
    km = np.clip(np.where(far, km + bump, km), 0.0, 15_000.0)

    frame = pd.DataFrame(
        {
            "amount": np.round(amount, 2),
            "hour_of_day": hour.astype(int),
            "day_of_week": dow.astype(int),
            "merchant_category": merchant.astype(int),
            "international": international,
            "billing_shipping_mismatch": mismatch,
            "txn_count_1h": txn_1h.astype(int),
            "txn_count_24h": txn_24h.astype(int),
            "mean_amount_7d": np.round(mean_amount, 2),
            "device_age_days": device_age.astype(int),
            "account_age_days": account_age.astype(int),
            "km_from_prev": np.round(km, 2),
        }
    )
    return frame[RAW_FIELDS]


def _draw_labels(rng: np.random.Generator, frame: pd.DataFrame) -> np.ndarray:
    model = to_model_frame(frame)
    weights = np.array([LABEL_WEIGHTS[name] for name in MODEL_FEATURES], dtype=np.float64)
    logit = model.to_numpy(dtype=np.float64) @ weights + LABEL_INTERCEPT
    logit = logit + rng.normal(0.0, LABEL_NOISE, size=len(frame))
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    return (rng.random(len(frame)) < probability).astype(np.int64)
