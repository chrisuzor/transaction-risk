"""Derived features shared by training and scoring."""

import numpy as np
import pandas as pd

from transaction_risk.constants import MODEL_FEATURES


def to_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    amount = frame["amount"].astype(float)
    mean_amount = frame["mean_amount_7d"].astype(float)
    hour = frame["hour_of_day"].astype(int)
    dow = frame["day_of_week"].astype(int)
    derived = {
        "log_amount": np.log1p(amount),
        "night": ((hour <= 5) | (hour >= 23)).astype(float),
        "weekend": (dow >= 5).astype(float),
        "merchant_risk": frame["merchant_category"].astype(int).isin([7, 8, 9]).astype(float),
        "international": frame["international"].astype(float),
        "billing_shipping_mismatch": frame["billing_shipping_mismatch"].astype(float),
        "txn_count_1h": frame["txn_count_1h"].astype(float),
        "txn_count_24h": frame["txn_count_24h"].astype(float),
        "amount_over_history": np.log1p(amount) - np.log1p(mean_amount),
        "device_age_days": frame["device_age_days"].astype(float),
        "account_age_days": frame["account_age_days"].astype(float),
        "log_km_from_prev": np.log1p(frame["km_from_prev"].astype(float)),
    }
    return pd.DataFrame(derived, index=frame.index)[MODEL_FEATURES]
