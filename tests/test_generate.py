import numpy as np
import pandas as pd

from transaction_risk.constants import N_ROWS, RAW_FIELDS, SEED
from transaction_risk.generate import generate_dataset, known_example
from transaction_risk.schema import Transaction


def test_generator_is_deterministic():
    frame_a, labels_a = generate_dataset(1000, 42)
    frame_b, labels_b = generate_dataset(1000, 42)
    pd.testing.assert_frame_equal(frame_a, frame_b)
    assert np.array_equal(labels_a, labels_b)


def test_different_seeds_differ():
    frame_a, _ = generate_dataset(200, 1)
    frame_b, _ = generate_dataset(200, 2)
    assert not frame_a.equals(frame_b)


def test_rows_are_synthetic_fields_only():
    frame, labels = generate_dataset(500, SEED)
    assert list(frame.columns) == RAW_FIELDS
    banned = {"card_number", "pan", "email", "name", "phone", "matric_no", "ssn"}
    assert banned.isdisjoint(frame.columns)
    assert set(np.unique(labels)).issubset({0, 1})
    assert frame["amount"].max() < 100_000
    assert frame["txn_count_24h"].ge(frame["txn_count_1h"]).all()


def test_known_example_validates():
    payload = known_example()
    Transaction.model_validate(payload)
    again = known_example()
    assert payload == again
    assert payload.keys() == set(RAW_FIELDS)


def test_default_dataset_size():
    frame, labels = generate_dataset(N_ROWS, SEED)
    assert len(frame) == N_ROWS
    assert len(labels) == N_ROWS
    rate = float(labels.mean())
    assert 0.03 <= rate <= 0.40
