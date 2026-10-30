from pathlib import Path

import numpy as np
import pandas as pd

from gaia_ai.config import load_config
from gaia_ai.liability import load_client, present_value_payments
from gaia_ai.risk import dynamic_risk_budget
from gaia_ai.sector import classify_regime, score_sectors


ROOT = Path(__file__).resolve().parents[1]


def test_regime_and_missing_revision_neutrality():
    assert classify_regime("slowing", "falling") == "Disinflationary Slowdown"
    assert classify_regime("rising", "rising") == "Overheat"
    sectors = pd.DataFrame({
        "sector": ["Utilities", "Real Estate"], "growth": [0.05, 0.08],
        "profitability": [0.2, 0.3], "valuation": [0.04, 0.03],
        "risk_adjusted_momentum": [-0.2, -0.4], "earnings_revisions": [np.nan, np.nan],
    })
    scored, metadata = score_sectors(sectors, "slowing", "falling", load_config().gaia["sector_weights"])
    assert scored["earnings_revisions_component"].eq(0.5).all()
    assert "not a complete GICS universe" in metadata["coverage_warning"]


def test_ahead_of_funding_path_harvests_risk_not_reverse():
    config = load_config().gaia["risk"]
    client = load_client(ROOT / "data/client_profile.json")
    market = {"spy_vol_percentile": 0.5, "portfolio_drawdown": 0.0}
    behind = dynamic_risk_budget(client, 2026, 100000, market, config)
    on_path = dynamic_risk_budget(client, 2026, present_value_payments(client, 2026), market, config)
    ahead = dynamic_risk_budget(client, 2026, 1500000, market, config)
    assert behind["growth_weight"] == on_path["growth_weight"]
    assert ahead["growth_weight"] < on_path["growth_weight"]
    assert np.isclose(sum(ahead["bucket_weights"].values()), 1)
