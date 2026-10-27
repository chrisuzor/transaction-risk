import pandas as pd

from transaction_risk.constants import MODEL_FEATURES
from transaction_risk.features import to_model_frame
from transaction_risk.generate import generate_dataset


def test_model_frame_columns():
    frame, _ = generate_dataset(10, 1)
    model = to_model_frame(frame)
    assert list(model.columns) == MODEL_FEATURES


def test_night_weekend_and_merchant_risk_flags():
    frame = pd.DataFrame(
        [
            {
                "amount": 10.0,
                "hour_of_day": 23,
                "day_of_week": 5,
                "merchant_category": 7,
                "international": False,
                "billing_shipping_mismatch": False,
                "txn_count_1h": 0,
                "txn_count_24h": 0,
                "mean_amount_7d": 10.0,
                "device_age_days": 10,
                "account_age_days": 10,
                "km_from_prev": 0.0,
            },
            {
                "amount": 10.0,
                "hour_of_day": 6,
                "day_of_week": 4,
                "merchant_category": 6,
                "international": False,
                "billing_shipping_mismatch": False,
                "txn_count_1h": 0,
                "txn_count_24h": 0,
                "mean_amount_7d": 10.0,
                "device_age_days": 10,
                "account_age_days": 10,
                "km_from_prev": 0.0,
            },
        ]
    )
    model = to_model_frame(frame)
    assert model.loc[0, "night"] == 1.0
    assert model.loc[0, "weekend"] == 1.0
    assert model.loc[0, "merchant_risk"] == 1.0
    assert model.loc[1, "night"] == 0.0
    assert model.loc[1, "weekend"] == 0.0
    assert model.loc[1, "merchant_risk"] == 0.0
