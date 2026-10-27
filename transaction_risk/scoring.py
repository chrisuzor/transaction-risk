"""Score raw transactions with the exported logistic regression."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from transaction_risk.constants import (
    ARTIFACT_NAME,
    MODEL_FEATURES,
    MODEL_VERSION,
    default_models_dir,
)
from transaction_risk.features import to_model_frame

_REQUIRED = (
    "model_version",
    "threshold",
    "feature_names",
    "scaler_mean",
    "scaler_scale",
    "weights",
    "bias",
)


def models_dir() -> Path:
    override = os.environ.get("TRANSACTION_RISK_MODELS")
    if override:
        return Path(override)
    return default_models_dir()


def artifact_path() -> Path:
    return models_dir() / ARTIFACT_NAME


def load_artifact(path: Path | None = None) -> dict:
    path = path or artifact_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model artifact at {path}. Run: python -m transaction_risk.train"
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in _REQUIRED if key not in artifact]
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
    if artifact["model_version"] != MODEL_VERSION:
        raise ValueError(
            f"artifact version is {artifact['model_version']}, expected {MODEL_VERSION}"
        )
    if list(artifact["feature_names"]) != MODEL_FEATURES:
        raise ValueError("artifact features do not match this code version")
    return artifact


def positive_scores(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    matrix = to_model_frame(frame).to_numpy(dtype=np.float64)
    mean = np.asarray(artifact["scaler_mean"], dtype=np.float64)
    scale = np.asarray(artifact["scaler_scale"], dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (matrix - mean) / scale
    logit = standardized @ np.asarray(artifact["weights"], dtype=np.float64)
    logit = logit + float(artifact["bias"])
    logit = np.clip(logit, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logit))


def score_payloads(artifact: dict, payloads: list[dict]) -> list[dict]:
    frame = pd.DataFrame(payloads)
    scores = positive_scores(artifact, frame)
    threshold = float(artifact["threshold"])
    version = str(artifact["model_version"])
    results = []
    for score in scores:
        value = float(score)
        results.append(
            {
                "score": value,
                "threshold": threshold,
                "decision": "flag" if value >= threshold else "allow",
                "model_version": version,
            }
        )
    return results
