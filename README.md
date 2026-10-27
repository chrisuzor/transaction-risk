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

## Out of scope

This slice stops at one trained baseline and a score API.

- No drift monitor
- No champion/challenger
- No live traffic
- No card numbers, customer identifiers, scraped records, or production metrics
- No threshold chosen from a monetary loss
- No payment authorization and no case-management workflow

## Development

```bash
pytest
```

The model card is `docs/model-card.md`. License: MIT.
