from pathlib import Path

import numpy as np
import pandas as pd

from gaia_ai.config import load_config
from gaia_ai.liability import load_client, present_value_payments
from gaia_ai.simulation import simulate_funding


ROOT = Path(__file__).resolve().parents[1]


def test_simulation_reproducible_and_sponsor_excluded_from_ofc():
    project = load_config()
    client = load_client(ROOT / "data/client_profile.json")
    weights = {"Growth": 0.85, "Stability": 0.15, "FutureLiability": 0.0}
    first = simulate_funding(client, 300306.83, weights, project.gaia["simulation"], paths=100)
    second = simulate_funding(client, 300306.83, weights, project.gaia["simulation"], paths=100)
    assert first["summary"]["operating_funding_confidence"] == second["summary"]["operating_funding_confidence"]
    pd.testing.assert_frame_equal(first["samples"], second["samples"])
    assert first["summary"]["sponsor_support_in_ofc"] is False
    assert 0 <= first["summary"]["operating_funding_confidence"] <= 1
    expected_reserve = present_value_payments(client, 2033) + client["flexibility_buffer"]
    assert np.isclose(first["summary"]["reserve_target_2033"], expected_reserve)
    assert len(first["payment_probability"]) == 10
    assert first["thresholds"]["target_confidence"].tolist() == [0.90, 0.95, 0.975, 0.99]
