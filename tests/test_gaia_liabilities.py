from pathlib import Path

import numpy as np

from gaia_ai.liability import commitment_range, load_client, payment_schedule, present_value_payments


ROOT = Path(__file__).resolve().parents[1]


def test_client_schedule_and_beginning_of_year_pv():
    client = load_client(ROOT / "data/client_profile.json")
    schedule = payment_schedule(client)
    assert schedule["year"].tolist() == list(range(2033, 2043))
    assert schedule["payment"].eq(50000).all()
    assert schedule["timing"].eq("beginning_of_year").all()
    expected = sum(50000 / (1.02 ** (year - 2033)) for year in range(2033, 2043))
    assert np.isclose(present_value_payments(client, 2033), expected)
    assert present_value_payments(client, 2043) == 0


def test_commitment_range_allows_zero_and_reports_tail():
    result = commitment_range(np.array([0, 0, 0, 10000, 20000, 40000]))
    assert result["recommended_lower"] >= 0
    assert result["recommended_upper"] >= result["recommended_lower"]
    assert result["p95"] >= result["p50"]
    assert 0 <= result["probability_zero_contribution"] <= 1
