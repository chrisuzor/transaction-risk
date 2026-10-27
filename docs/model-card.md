# Model card: baseline-v1

Logistic regression for synthetic card-not-present transaction risk.

Author: Christopher Uzochukwu. Timestamps use Africa/Lagos.

## Data

Every row is synthetic. `transaction_risk.generate` draws 20,000 rows with seed 42 from fixed random distributions. Labels are a Bernoulli draw from a noisy linear function of the model features. The label weights are in the generator and are not copied into this card.

No real cardholder, card number, merchant name, scraped table, or production metric is used.

The split is stratified: 70% train, 15% validation, 15% test. The threshold is chosen on validation. Precision, recall, false positives, and false negatives below are from the test split only.

## Features

Raw fields accepted by the API: `amount`, `hour_of_day`, `day_of_week`, `merchant_category` (synthetic id 0–9), `international`, `billing_shipping_mismatch`, `txn_count_1h`, `txn_count_24h`, `mean_amount_7d`, `device_age_days`, `account_age_days`, `km_from_prev`.

Derived model features: `log_amount`, `night` (hour 0–5 or 23), `weekend` (day index 5 or 6), `merchant_risk` (category 7, 8, or 9), `international`, `billing_shipping_mismatch`, `txn_count_1h`, `txn_count_24h`, `amount_over_history`, `device_age_days`, `account_age_days`, `log_km_from_prev`.

## Metric at the chosen threshold

<!-- metrics:start -->
Trained at 2026-10-27T01:35:57+01:00 (Africa/Lagos).

Held-out evaluation from `python -m transaction_risk.train` on 27 October 2026 (Africa/Lagos), trained at 2026-10-27T01:35:57+01:00.

On the held-out test split, at threshold 0.112: precision is 0.3558 and recall is 0.8803, with 985 false positives and 74 false negatives. True positives: 544. True negatives: 1397.

Threshold rule: highest threshold on a 0.001 grid with validation recall >= 0.90. Validation recall at this threshold is 0.9045. The synthetic positive rate on the test split is 0.2060.

Fitted coefficients are on standardized features (training-set mean and scale):

| Feature | Coefficient |
| --- | --- |
| log_amount | 0.3083 |
| night | 0.5259 |
| weekend | 0.0745 |
| merchant_risk | 0.3792 |
| international | 0.4343 |
| billing_shipping_mismatch | 0.3935 |
| txn_count_1h | 0.3327 |
| txn_count_24h | 0.0274 |
| amount_over_history | 0.8319 |
| device_age_days | -0.4939 |
| account_age_days | -0.3405 |
| log_km_from_prev | 0.4528 |

Intercept after scaling: -1.9446.

Library versions at training: scikit-learn 1.9.1, numpy 2.5.4.

Training script output:

```
held-out evaluation
trained_at=2026-10-27T01:35:57+01:00
timezone=Africa/Lagos
model_version=baseline-v1
data=synthetic
channel=card_not_present
seed=42
n_rows=20000
n_train=14000
n_validation=3000
n_test=3000
synthetic_positive_rate=0.2059
threshold_rule=highest threshold on a 0.001 grid with validation recall >= 0.90
threshold=0.112
precision=0.3558
recall=0.8803
false_positives=985
false_negatives=74
true_positives=544
true_negatives=1397
validation_recall=0.9045
test_positive_rate=0.2060
sklearn=1.9.1
numpy=2.5.4
```
<!-- metrics:end -->

## Threshold

Recall-oriented. The rule is the highest threshold on a 0.001 grid with validation recall >= 0.90. The decision is `flag` when the score is greater than or equal to that threshold, and `allow` otherwise. The choice keeps more synthetic fraud labels and accepts more false positives than a 0.50 cutoff. No loss amount is used.

## Limits

- The fit is to the synthetic label process. These metrics are not field performance on real payments.
- One linear model, with the derived features above. One frozen threshold.
- No drift monitor, no champion/challenger, and no live traffic.
- Card-not-present rows only.
- The score is not a payment-decline reason. This service does not authorize payments.
- Coefficients are in standardized units (training-set mean and scale).

## Artifact

`models/baseline-v1.json`, version `baseline-v1`. Rebuild with `python -m transaction_risk.train --write-model-card`.
