from pathlib import Path

import numpy as np

from gaia_ai.config import load_config
from gaia_ai.pipeline import run_full_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_offline_is_research_only(tmp_path):
    result = run_full_pipeline(
        load_config(ROOT / "config/default.json"), ROOT / "data", tmp_path,
        as_of="2026-10-29", narrative_provider="extractive", simulation_paths=50,
    )
    summary = result["summary"]
    assert summary["system_status"] == "OFFLINE_DEMO_RESEARCH_ONLY_NO_TRADE"
    assert summary["narrative_provider_is_llm"] is False
    assert summary["human_approved_candidates"] == 0
    assert summary["allocation_eligible_candidates"] == 0
    assert len(result["frames"]["narrative_agents"]) == 60
    assert len(result["frames"]["narrative_evidence"]) == 480
    assert result["frames"]["allocation_audit"]["passed"].all()
    positions = result["frames"]["allocation_research"]
    stocks = positions[positions["代码"] != "SPYM"]
    assert np.isclose(stocks["research_weight"].sum(), 0.40)
    assert np.isclose(stocks["approved_target_weight"].sum(), 0.0)
    assert np.isclose(positions["research_weight"].sum(), 1.0)
    assert (tmp_path / "GAIA_full_system_report.xlsx").exists()
    assert (tmp_path / "full_system_summary.json").exists()
