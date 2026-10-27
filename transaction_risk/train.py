"""Train the baseline-v1 logistic regression and print held-out metrics.

The script fits on synthetic rows only, chooses a recall-oriented threshold
on the validation split, and reports precision, recall, false positives, and
false negatives on the untouched test split.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transaction_risk.constants import (
    ARTIFACT_NAME,
    CHANNEL,
    EVAL_NAME,
    KNOWN_EXAMPLE_NAME,
    MODEL_FEATURES,
    MODEL_VERSION,
    N_ROWS,
    SEED,
    TARGET_RECALL,
    THRESHOLD_RULE,
    TIMEZONE,
    default_models_dir,
    repo_root,
)
from transaction_risk.features import to_model_frame
from transaction_risk.generate import generate_dataset, known_example, row_to_payload
from transaction_risk.schema import Transaction
from transaction_risk.scoring import positive_scores
from transaction_risk.threshold import choose_threshold

import sklearn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the synthetic CNP risk baseline")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-rows", type=int, default=N_ROWS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--write-model-card",
        action="store_true",
        help="Refresh metric blocks in README.md and docs/model-card.md",
    )
    args = parser.parse_args(argv)
    if args.n_rows < 1000:
        raise SystemExit("n-rows must be at least 1000")

    output_dir = args.output_dir or default_models_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, labels = generate_dataset(args.n_rows, args.seed)
    example = row_to_payload(frame.iloc[0])
    Transaction.model_validate(example)
    if args.seed == SEED and args.n_rows == N_ROWS and example != known_example():
        raise SystemExit("row 0 drifted from known_example()")
    overall_rate = float(labels.mean())
    if not 0.03 <= overall_rate <= 0.40:
        raise SystemExit(
            f"synthetic positive rate {overall_rate:.4f} is outside 0.03-0.40"
        )

    raw_train, raw_temp, y_train, y_temp = train_test_split(
        frame,
        labels,
        test_size=0.30,
        random_state=args.seed,
        stratify=labels,
    )
    raw_val, raw_test, y_val, y_test = train_test_split(
        raw_temp,
        y_temp,
        test_size=0.50,
        random_state=args.seed,
        stratify=y_temp,
    )
    for name, part in (("train", y_train), ("validation", y_val), ("test", y_test)):
        if len(np.unique(part)) < 2:
            raise SystemExit(f"{name} split is single-class")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=1000,
                    tol=1e-8,
                ),
            ),
        ]
    )
    x_train = to_model_frame(raw_train).to_numpy(dtype=np.float64)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    if clf.classes_.tolist() != [0, 1]:
        raise SystemExit(f"unexpected classes: {clf.classes_.tolist()}")

    artifact = {
        "model_version": MODEL_VERSION,
        "model_type": "logistic_regression",
        "data": "synthetic",
        "channel": CHANNEL,
        "seed": args.seed,
        "n_rows": args.n_rows,
        "threshold_rule": THRESHOLD_RULE,
        "target_recall": TARGET_RECALL,
        "feature_names": list(MODEL_FEATURES),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "weights": [float(value) for value in clf.coef_.ravel()],
        "bias": float(clf.intercept_[0]),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    if len(artifact["weights"]) != len(MODEL_FEATURES):
        raise SystemExit("weight vector does not match feature list")

    # Placeholder so scoring can run before the threshold is known.
    artifact["threshold"] = 0.5
    for split_name, split_frame in (
        ("train", raw_train),
        ("validation", raw_val),
        ("test", raw_test),
    ):
        matrix = to_model_frame(split_frame).to_numpy(dtype=np.float64)
        sklearn_scores = model.predict_proba(matrix)[:, 1]
        manual_scores = positive_scores(artifact, split_frame)
        gap = float(np.max(np.abs(sklearn_scores - manual_scores)))
        if gap > 1e-6:
            raise SystemExit(f"scorer drifted from sklearn on {split_name}: max abs {gap}")

    val_scores = positive_scores(artifact, raw_val)
    threshold = choose_threshold(y_val, val_scores, TARGET_RECALL)
    artifact["threshold"] = threshold
    trained_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")
    artifact["trained_at"] = trained_at
    artifact["timezone"] = TIMEZONE

    test_scores = positive_scores(artifact, raw_test)
    counts = _counts(y_test, test_scores, threshold)
    val_counts = _counts(y_val, val_scores, threshold)
    if counts["false_positives"] == 0 or counts["false_negatives"] == 0:
        raise SystemExit("held-out false positives and false negatives must both be non-zero")
    if counts["precision"] <= counts["positive_rate"]:
        raise SystemExit("precision is not above the synthetic test positive rate")

    lines = [
        "held-out evaluation",
        f"trained_at={trained_at}",
        f"timezone={TIMEZONE}",
        f"model_version={MODEL_VERSION}",
        "data=synthetic",
        f"channel={CHANNEL}",
        f"seed={args.seed}",
        f"n_rows={args.n_rows}",
        f"n_train={len(y_train)}",
        f"n_validation={len(y_val)}",
        f"n_test={len(y_test)}",
        f"synthetic_positive_rate={overall_rate:.4f}",
        f"threshold_rule={THRESHOLD_RULE}",
        f"threshold={threshold:.3f}",
        f"precision={counts['precision']:.4f}",
        f"recall={counts['recall']:.4f}",
        f"false_positives={counts['false_positives']}",
        f"false_negatives={counts['false_negatives']}",
        f"true_positives={counts['true_positives']}",
        f"true_negatives={counts['true_negatives']}",
        f"validation_recall={val_counts['recall']:.4f}",
        f"test_positive_rate={counts['positive_rate']:.4f}",
        f"sklearn={sklearn.__version__}",
        f"numpy={np.__version__}",
    ]
    stdout = "\n".join(lines)
    print(stdout)

    evaluation = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "timezone": TIMEZONE,
        "data": "synthetic",
        "channel": CHANNEL,
        "seed": args.seed,
        "n_rows": args.n_rows,
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "synthetic_positive_rate": overall_rate,
        "threshold_rule": THRESHOLD_RULE,
        "target_recall": TARGET_RECALL,
        "threshold": threshold,
        "precision": counts["precision"],
        "recall": counts["recall"],
        "false_positives": counts["false_positives"],
        "false_negatives": counts["false_negatives"],
        "true_positives": counts["true_positives"],
        "true_negatives": counts["true_negatives"],
        "validation_recall": val_counts["recall"],
        "test_positive_rate": counts["positive_rate"],
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "stdout": stdout,
    }
    _write_json(output_dir / ARTIFACT_NAME, artifact)
    _write_json(output_dir / EVAL_NAME, evaluation)
    _write_json(output_dir / KNOWN_EXAMPLE_NAME, example)
    print(f"artifact={_display_path(output_dir / ARTIFACT_NAME)}")
    print(f"eval={_display_path(output_dir / EVAL_NAME)}")
    print(f"known_example={_display_path(output_dir / KNOWN_EXAMPLE_NAME)}")

    if args.write_model_card:
        _update_docs(evaluation, artifact)
        print("updated README.md and docs/model-card.md")


def _counts(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(y_true).astype(bool)
    flagged = np.asarray(scores) >= threshold
    true_positives = int(np.sum(flagged & labels))
    false_positives = int(np.sum(flagged & ~labels))
    false_negatives = int(np.sum(~flagged & labels))
    true_negatives = int(np.sum(~flagged & ~labels))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "positive_rate": float(labels.mean()),
    }


def _update_docs(evaluation: dict, artifact: dict) -> None:
    trained = datetime.fromisoformat(evaluation["trained_at"])
    when = f"{trained.day} {trained.strftime('%B')} {trained.year}"
    prose = (
        f"Held-out evaluation from `python -m transaction_risk.train` "
        f"on {when} (Africa/Lagos), trained at {evaluation['trained_at']}.\n\n"
        f"On the held-out test split, at threshold {evaluation['threshold']:.3f}: "
        f"precision is {evaluation['precision']:.4f} and recall is {evaluation['recall']:.4f}, "
        f"with {evaluation['false_positives']} false positives and "
        f"{evaluation['false_negatives']} false negatives. "
        f"True positives: {evaluation['true_positives']}. "
        f"True negatives: {evaluation['true_negatives']}.\n\n"
        f"Threshold rule: {evaluation['threshold_rule']}. "
        f"Validation recall at this threshold is {evaluation['validation_recall']:.4f}. "
        f"The synthetic positive rate on the test split is {evaluation['test_positive_rate']:.4f}."
    )
    coefficient_rows = "\n".join(
        f"| {name} | {weight:.4f} |"
        for name, weight in zip(artifact["feature_names"], artifact["weights"], strict=True)
    )
    card = (
        f"Trained at {evaluation['trained_at']} (Africa/Lagos).\n\n"
        f"{prose}\n\n"
        "Fitted coefficients are on standardized features "
        "(training-set mean and scale):\n\n"
        "| Feature | Coefficient |\n"
        "| --- | --- |\n"
        f"{coefficient_rows}\n\n"
        f"Intercept after scaling: {artifact['bias']:.4f}.\n\n"
        f"Library versions at training: scikit-learn {evaluation['sklearn_version']}, "
        f"numpy {evaluation['numpy_version']}.\n\n"
        "Training script output:\n\n"
        "```\n"
        f"{evaluation['stdout']}\n"
        "```"
    )
    _replace_block(repo_root() / "README.md", "<!-- eval:start -->", "<!-- eval:end -->", prose)
    _replace_block(
        repo_root() / "docs" / "model-card.md",
        "<!-- metrics:start -->",
        "<!-- metrics:end -->",
        card,
    )


def _replace_block(path: Path, start: str, end: str, body: str) -> None:
    content = path.read_text(encoding="utf-8")
    if start not in content or end not in content:
        raise SystemExit(f"{path} is missing marker {start}")
    pre, rest = content.split(start, 1)
    _, post = rest.split(end, 1)
    path.write_text(pre + start + "\n" + body.rstrip() + "\n" + end + post, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
