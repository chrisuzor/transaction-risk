import json

from transaction_risk.constants import (
    EVAL_NAME,
    THRESHOLD_RULE,
    default_models_dir,
    repo_root,
)


def test_readme_covers_the_slice():
    readme = (repo_root() / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## Problem",
        "## Features",
        "## How to train",
        "## How to run",
        "## Threshold trade-off",
        "## Drift check",
        "## Champion and challenger",
        "## Out of scope",
    ):
        assert heading in readme
    lowered = readme.lower()
    assert "every row is synthetic" in lowered
    assert "drift monitor" in lowered
    assert "champion/challenger" in lowered
    assert "live traffic" in lowered
    assert THRESHOLD_RULE in readme
    assert "africa/lagos" in lowered


def test_published_metrics_match_eval_json():
    evaluation = json.loads(
        (default_models_dir() / EVAL_NAME).read_text(encoding="utf-8")
    )
    phrase = (
        f"precision is {evaluation['precision']:.4f} and recall is {evaluation['recall']:.4f}, "
        f"with {evaluation['false_positives']} false positives and "
        f"{evaluation['false_negatives']} false negatives"
    )
    readme = (repo_root() / "README.md").read_text(encoding="utf-8")
    card = (repo_root() / "docs" / "model-card.md").read_text(encoding="utf-8")
    assert phrase in readme
    assert phrase in card
    assert evaluation["stdout"] in card
    assert evaluation["trained_at"] in card
    assert "synthetic" in card.lower()
    assert "both windows are synthetic" in readme.lower()
    assert "both windows are synthetic" in card.lower()
    assert "score_psi=0.0030" in readme
    assert "score_psi=12.4338" in readme
    assert "score_psi=0.0030" in card
    assert "score_psi=12.4338" in card
    assert "below 0.25" in readme
    assert "below 0.25" in card
    for line in (
        "agreement_rate=0.6087",
        "disagreement_count=1174",
        "champion_false_positives=985",
        "champion_false_negatives=74",
        "challenger_false_positives=118",
        "challenger_false_negatives=381",
    ):
        assert line in readme
        assert line in card
    assert "shadow" in readme.lower()
    assert "second threshold" in readme.lower()
