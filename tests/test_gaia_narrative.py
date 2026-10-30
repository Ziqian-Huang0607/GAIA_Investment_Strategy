import pandas as pd
import pytest

from gaia_ai.config import load_config
from gaia_ai.narrative import DIMENSIONS, ExtractiveNarrativeProvider, NarrativeEngine


def document():
    return pd.DataFrame([{
        "ticker": "DEMO", "document_id": "demo-1", "document_date": "2026-10-29",
        "document_type": "team_research_demo", "text": "Demand and organic growth improved. Pricing power supported margin. Management execution was strong. Debt and capex remain risks.",
        "source_url": "https://example.test/demo", "is_demo": True,
    }])


def test_five_agents_evidence_and_offline_gate():
    config = load_config().gaia
    engine = NarrativeEngine(config["agents"], config["uncertainty"]["min_ai_confidence"], config["uncertainty"]["max_ads"])
    evidence, agents, summary = engine.run(document(), ExtractiveNarrativeProvider())
    assert len(agents) == 5
    assert len(evidence) == 5 * 8
    assert agents["agent"].nunique() == 5
    assert evidence["source_url"].eq("https://example.test/demo").all()
    assert summary.loc[0, "ai_gate"] == "REVIEW_REQUIRED"
    assert "OFFLINE_EXTRACTIVE_NOT_LLM" in summary.loc[0, "gate_reasons"]
    assert "DEMO_DOCUMENTS" in summary.loc[0, "gate_reasons"]
    assert summary.loc[0, "ads"] >= 0


def test_llm_evidence_must_be_exact_quote():
    class BadProvider:
        name = "openai_compatible_llm"
        def analyze(self, doc, role):
            return {
                "overall_score": 3, "overall_confidence": 0.8, "rationale": "test",
                "dimensions": [{
                    "dimension": name, "score": 3, "confidence": 0.8,
                    "direction_of_change": 0, "evidence": "invented quote",
                } for name in DIMENSIONS],
            }
    config = load_config().gaia
    engine = NarrativeEngine(config["agents"], 0.6, 0.75)
    with pytest.raises(ValueError, match="exact quote"):
        engine.run(document(), BadProvider())
