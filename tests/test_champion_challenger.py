import json

from transaction_risk.champion_challenger import compare_fixed_batch
from transaction_risk.constants import (
    ARTIFACT_NAME,
    CHALLENGER_THRESHOLD,
    EVAL_NAME,
    default_models_dir,
)
from transaction_risk.scoring import load_artifact


def _committed_report() -> dict:
    artifact = load_artifact(default_models_dir() / ARTIFACT_NAME)
    return compare_fixed_batch(artifact)


# Locked to the fixed seed-42 test split and the committed champion artifact.
EXPECTED = {
    "n_batch": 3000,
    "agreement_count": 1826,
    "disagreement_count": 1174,
    "champion_false_positives": 985,
    "champion_false_negatives": 74,
    "challenger_false_positives": 118,
    "challenger_false_negatives": 381,
}


def test_fixed_seed_counts_are_locked():
    report = _committed_report()
    for key, expected in EXPECTED.items():
        assert report[key] == expected
    assert f"{report['agreement_rate']:.4f}" == "0.6087"
    committed = load_artifact(default_models_dir() / ARTIFACT_NAME)
    assert report["champion_threshold"] == committed["threshold"]
    assert report["challenger_threshold"] == CHALLENGER_THRESHOLD


def test_champion_counts_match_published_held_out_eval():
    evaluation = json.loads((default_models_dir() / EVAL_NAME).read_text(encoding="utf-8"))
    report = _committed_report()
    assert report["champion_false_positives"] == evaluation["false_positives"]
    assert report["champion_false_negatives"] == evaluation["false_negatives"]


def test_challenger_is_shadow_only_and_records_disagreement():
    report = _committed_report()
    assert len(report["records"]) == EXPECTED["n_batch"]
    disagreements = 0
    for row in report["records"]:
        assert row["served_decision"] == row["champion_decision"]
        assert row["disagree"] is (row["champion_decision"] != row["challenger_decision"])
        assert set(row) >= {
            "champion_score",
            "champion_decision",
            "challenger_score",
            "challenger_decision",
            "disagree",
        }
        if row["disagree"]:
            disagreements += 1
            assert row["champion_decision"] == "flag"
            assert row["challenger_decision"] == "allow"
    assert disagreements == EXPECTED["disagreement_count"]


def test_false_counts_match_the_recorded_decisions():
    report = _committed_report()
    champion_fp = _count(report, "champion_decision", "flag", 0)
    champion_fn = _count(report, "champion_decision", "allow", 1)
    challenger_fp = _count(report, "challenger_decision", "flag", 0)
    challenger_fn = _count(report, "challenger_decision", "allow", 1)
    assert champion_fp == EXPECTED["champion_false_positives"]
    assert champion_fn == EXPECTED["champion_false_negatives"]
    assert challenger_fp == EXPECTED["challenger_false_positives"]
    assert challenger_fn == EXPECTED["challenger_false_negatives"]


def _count(report: dict, decision_key: str, decision: str, label: int) -> int:
    return sum(
        1
        for row in report["records"]
        if row[decision_key] == decision and row["label"] == label
    )
