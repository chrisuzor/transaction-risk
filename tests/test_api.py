import json

import pytest
from fastapi.testclient import TestClient

from transaction_risk.api import app
from transaction_risk.constants import EVAL_NAME, KNOWN_EXAMPLE_NAME, MODEL_VERSION
from transaction_risk.generate import known_example, payloads_at
from transaction_risk.scoring import load_artifact, models_dir, score_payloads


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_score_known_synthetic_example(client):
    payload = known_example()
    on_disk = json.loads((models_dir() / KNOWN_EXAMPLE_NAME).read_text(encoding="utf-8"))
    assert payload == on_disk
    expected = score_payloads(load_artifact(), [payload])[0]
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == MODEL_VERSION
    assert body["score"] == pytest.approx(expected["score"], abs=1e-9)
    assert body["threshold"] == pytest.approx(expected["threshold"])
    assert body["decision"] == expected["decision"]
    assert body["decision"] == ("flag" if body["score"] >= body["threshold"] else "allow")


def test_score_batch_preserves_order(client):
    rows = payloads_at([0, 1])
    assert rows[0] == known_example()
    expected = score_payloads(load_artifact(), rows)
    response = client.post("/v1/score/batch", json={"transactions": rows})
    assert response.status_code == 200
    body = response.json()["results"]
    assert len(body) == 2
    for got, exp in zip(body, expected, strict=True):
        assert got["score"] == pytest.approx(exp["score"], abs=1e-9)
        assert got["decision"] == exp["decision"]
        assert got["threshold"] == pytest.approx(exp["threshold"])
        assert got["model_version"] == MODEL_VERSION


def test_batch_contains_both_decisions(client):
    rows = payloads_at(list(range(400)))
    scored = score_payloads(load_artifact(), rows)
    flag_row = next(rows[i] for i, item in enumerate(scored) if item["decision"] == "flag")
    allow_row = next(rows[i] for i, item in enumerate(scored) if item["decision"] == "allow")
    response = client.post(
        "/v1/score/batch",
        json={"transactions": [flag_row, allow_row]},
    )
    assert response.status_code == 200
    decisions = [item["decision"] for item in response.json()["results"]]
    assert decisions == ["flag", "allow"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == MODEL_VERSION


def test_closed_schema_rejects_card_number(client):
    payload = known_example()
    payload["card_number"] = "not-accepted"
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422


def test_batch_limits(client):
    empty = client.post("/v1/score/batch", json={"transactions": []})
    assert empty.status_code == 422
    too_many = client.post(
        "/v1/score/batch",
        json={"transactions": [known_example() for _ in range(51)]},
    )
    assert too_many.status_code == 422


def test_inconsistent_velocity_is_rejected(client):
    payload = known_example()
    payload["txn_count_1h"] = 4
    payload["txn_count_24h"] = 1
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422


def test_artifact_matches_eval_record():
    directory = models_dir()
    artifact = load_artifact(directory / "baseline-v1.json")
    evaluation = json.loads((directory / EVAL_NAME).read_text(encoding="utf-8"))
    assert artifact["threshold"] == evaluation["threshold"]
    assert artifact["model_version"] == evaluation["model_version"] == MODEL_VERSION
    assert evaluation["validation_recall"] >= 0.90
    assert evaluation["data"] == "synthetic"
