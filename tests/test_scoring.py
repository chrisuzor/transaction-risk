import pytest

from transaction_risk.constants import MODEL_FEATURES, MODEL_VERSION
from transaction_risk.generate import known_example
from transaction_risk.scoring import score_payloads


def test_zero_weights_score_at_half_and_flag_on_the_boundary():
    artifact = {
        "model_version": MODEL_VERSION,
        "threshold": 0.5,
        "feature_names": list(MODEL_FEATURES),
        "scaler_mean": [0.0] * len(MODEL_FEATURES),
        "scaler_scale": [1.0] * len(MODEL_FEATURES),
        "weights": [0.0] * len(MODEL_FEATURES),
        "bias": 0.0,
    }
    scored = score_payloads(artifact, [known_example()])[0]
    assert scored["score"] == pytest.approx(0.5, abs=1e-12)
    assert scored["decision"] == "flag"
    assert scored["model_version"] == MODEL_VERSION
