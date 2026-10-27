# Transaction risk

[![ci](https://github.com/chrisuzor/transaction-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisuzor/transaction-risk/actions/workflows/ci.yml)

Synthetic card-not-present risk scores. **Every row is synthetic.** The generator draws each field from a fixed-seed random distribution. This repository has no real cardholder, no real card number, no scraped table, no production metric, and no employer data.

Author: Christopher Uzochukwu. Timestamps use Africa/Lagos.

## Problem

Card-not-present payments are authorized without the physical card, so a risk score is often computed before a decision. This repository is a small public baseline of that shape: a fixed-seed generator, a logistic regression, one frozen threshold, and an HTTP API that returns a score plus the threshold decision.

## Features

The API accepts one raw transaction. Categories are synthetic ids, not card-network merchant category codes.

| Field | Meaning |
| --- | --- |
| `amount` | Synthetic currency units |
| `hour_of_day` | Integer 0–23 |
| `day_of_week` | Synthetic day index 0–6 |
| `merchant_category` | Integer 0–9 |
| `international` | Cross-border flag |
| `billing_shipping_mismatch` | Billing and shipping disagree |
| `txn_count_1h`, `txn_count_24h` | Recent velocity. The 24-hour count is at least the 1-hour count |
| `mean_amount_7d` | Recent average amount |
| `device_age_days`, `account_age_days` | Age of the synthetic device and account |
| `km_from_prev` | Distance from the previous synthetic event |

The model sees derived features: `log_amount`, `night` (hour 0–5 or 23), `weekend` (day 5 or 6), `merchant_risk` (category 7, 8, or 9), the two flags, both velocity counts, `amount_over_history` (`log1p(amount) - log1p(mean_amount_7d)`), both ages, and `log_km_from_prev`.

Labels are a noisy linear function of those derived features. The weights live only in `transaction_risk/generate.py`. The trainer does not read them.

## How to train

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m transaction_risk.train --write-model-card
```

The default run uses seed `42` and `20_000` rows. The split is stratified: 70% train, 15% validation, 15% test. Validation is used only to freeze the threshold. The printed precision, recall, false positives, and false negatives are the test split.

Training writes:

- `models/baseline-v1.json` — versioned logistic regression (standardized coefficients, bias, threshold)
- `models/eval.json` — held-out counts and the script output
- `models/known_example.json` — row 0 of the default generator

The API scores with the JSON coefficients. It does not unpickle a model and it does not download a pretrained model. GitHub Actions reruns this script into a temporary directory, then pytest loads the committed artifact.

## How to run

```bash
uvicorn transaction_risk.api:app --host 0.0.0.0 --port 8000
```

```bash
docker build -t transaction-risk .
docker run --rm -p 8000:8000 transaction-risk
```

The image loads `models/baseline-v1.json` and starts the API. It does not train on startup.

`POST /v1/score` takes one transaction. `POST /v1/score/batch` takes `{"transactions": [ ... ]}` with 1 to 50 rows. Both return `score`, `threshold`, `decision` (`flag` when `score >= threshold`, otherwise `allow`), and `model_version`.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/score \
  -H 'content-type: application/json' \
  --data-binary @models/known_example.json
```

`GET /health` returns the loaded model version.

## Threshold trade-off

The operating point is recall-oriented. On the validation split the trainer keeps the highest threshold on a 0.001 grid with validation recall >= 0.90. That is the strictest cutoff that still catches at least 90% of the synthetic fraud labels in validation. A lower threshold would flag more rows and add false positives. A higher threshold would miss more synthetic fraud labels. No monetary loss, chargeback amount, or review cost is used.

`highest threshold on a 0.001 grid with validation recall >= 0.90`

<!-- eval:start -->
Held-out evaluation from `python -m transaction_risk.train` on 27 October 2026 (Africa/Lagos), trained at 2026-10-27T01:35:57+01:00.

On the held-out test split, at threshold 0.112: precision is 0.3558 and recall is 0.8803, with 985 false positives and 74 false negatives. True positives: 544. True negatives: 1397.

Threshold rule: highest threshold on a 0.001 grid with validation recall >= 0.90. Validation recall at this threshold is 0.9045. The synthetic positive rate on the test split is 0.2060.
<!-- eval:end -->

## Drift check

Both windows are synthetic. The reference is the training split of `generate_dataset(n=20000, seed=42)`. The unshifted holdout is the test split of that same draw. The shifted window is `generate_shifted_dataset(n=3000, seed=7)`. Its amounts, velocity, distance, ages, and flags are drawn from different fixed distributions than training.

The statistic is the population stability index, reference against the comparison window. A feature with at most 10 distinct reference values uses one bin per value. Other features use 10 quantile bins from the reference. A window is stable when every model-feature PSI and the score PSI are below 0.25. A value at or above 0.25 is not stable. The 0.25 cutoff is the usual significant-shift mark for PSI. It is not a loss amount.

```bash
python -m transaction_risk.drift
```

Checked at 2026-10-27T01:45:15+01:00 (Africa/Lagos). Unshifted holdout: `stable=true`, `score_psi=0.0030`. Shifted window: `stable=false`, `score_psi=12.4338`.

| Feature | Unshifted PSI | Shifted PSI |
| --- | ---: | ---: |
| log_amount | 0.0030 | 6.2854 |
| night | 0.0018 | 10.4266 |
| weekend | 0.0008 | 10.4954 |
| merchant_risk | 0.0000 | 10.1985 |
| international | 0.0000 | 1.5806 |
| billing_shipping_mismatch | 0.0004 | 0.9126 |
| txn_count_1h | 0.0021 | 6.5627 |
| txn_count_24h | 0.0017 | 6.3973 |
| amount_over_history | 0.0031 | 9.5442 |
| device_age_days | 0.0047 | 12.4355 |
| account_age_days | 0.0037 | 12.4372 |
| log_km_from_prev | 0.0047 | 3.3246 |
| score | 0.0030 | 12.4338 |

## Champion and challenger

The champion is `baseline-v1`, the logistic regression already in the repo. Its threshold decision is the one that would be served. The challenger is a second threshold on those same scores, not a second algorithm.

A gradient boosting model trained on the same seed and the same split was a worse fit. The synthetic labels are a noisy linear function of the model features, so logistic regression is the matched model. On the test split the boosting fit had higher log loss (0.389 versus 0.374), a higher Brier score (0.122 versus 0.118), and more false positives at the same validation-recall rule (1067 versus 985). The shadow threshold is 0.50. It is the conventional probability cutoff, stricter than the champion's recall-oriented 0.112, and it is not chosen from a loss amount.

The batch is the synthetic test split of `generate_dataset(n=20000, seed=42)`. For each row the script records the champion score, the champion decision, the challenger score, the challenger decision, and whether they disagree. The challenger is shadow-only.

```bash
python -m transaction_risk.champion_challenger
```

Checked at 2026-10-27T01:50:25+01:00 (Africa/Lagos):

```
agreement_rate=0.6087
disagreement_count=1174
champion_false_positives=985
champion_false_negatives=74
challenger_false_positives=118
challenger_false_negatives=381
```

`n_batch=3000` and `agreement_count=1826`. Every disagreement on this batch is a champion `flag` with a challenger `allow`.

## Out of scope

- No live drift monitor. The drift section is an offline check on synthetic windows.
- No live champion/challenger. The comparison above is an offline shadow on one synthetic batch.
- No live traffic
- No card numbers, customer identifiers, scraped records, or production metrics
- No threshold chosen from a monetary loss
- No payment authorization and no case-management workflow

## Development

```bash
pytest
```

The model card is `docs/model-card.md`. License: MIT.
