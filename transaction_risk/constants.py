"""Shared names for the baseline-v1 slice."""

from pathlib import Path

MODEL_VERSION = "baseline-v1"
SEED = 42
N_ROWS = 20_000
TARGET_RECALL = 0.90
TIMEZONE = "Africa/Lagos"
CHANNEL = "card_not_present"

THRESHOLD_RULE = (
    "highest threshold on a 0.001 grid with validation recall >= "
    f"{TARGET_RECALL:.2f}"
)

ARTIFACT_NAME = "baseline-v1.json"
EVAL_NAME = "eval.json"
KNOWN_EXAMPLE_NAME = "known_example.json"
KNOWN_EXAMPLE_INDEX = 0

RAW_FIELDS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category",
    "international",
    "billing_shipping_mismatch",
    "txn_count_1h",
    "txn_count_24h",
    "mean_amount_7d",
    "device_age_days",
    "account_age_days",
    "km_from_prev",
]

MODEL_FEATURES = [
    "log_amount",
    "night",
    "weekend",
    "merchant_risk",
    "international",
    "billing_shipping_mismatch",
    "txn_count_1h",
    "txn_count_24h",
    "amount_over_history",
    "device_age_days",
    "account_age_days",
    "log_km_from_prev",
]

# Label-process weights. Used only to draw synthetic labels.
LABEL_INTERCEPT = -4.6
LABEL_NOISE = 0.85
LABEL_WEIGHTS = {
    "log_amount": 0.45,
    "night": 1.20,
    "weekend": 0.20,
    "merchant_risk": 0.90,
    "international": 1.50,
    "billing_shipping_mismatch": 1.70,
    "txn_count_1h": 0.55,
    "txn_count_24h": 0.05,
    "amount_over_history": 1.10,
    "device_age_days": -0.0012,
    "account_age_days": -0.0005,
    "log_km_from_prev": 0.30,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_models_dir() -> Path:
    return repo_root() / "models"
