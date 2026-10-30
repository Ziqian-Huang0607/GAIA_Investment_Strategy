from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .liability import liability_phase


RESEARCH_CONFIDENCE = {
    "high": 0.95,
    "medium-high": 0.80,
    "medium": 0.65,
    "medium-low": 0.50,
    "low": 0.35,
}


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _risk_aversion(year: int) -> float:
    return {
        "Growth": 0.20,
        "Transition": 0.55,
        "Protection": 0.90,
        "Liability Segmentation": 1.00,
    }[liability_phase(year)]


def build_conviction(
    prepared: pd.DataFrame,
    score_quarter: pd.Timestamp,
    predictions: pd.DataFrame,
    human_research: pd.DataFrame,
    narrative_summary: pd.DataFrame,
    sector_scores: pd.DataFrame,
    year: int,
    gaia_config: dict,
) -> pd.DataFrame:
    current = prepared[prepared["季度"] == score_quarter].copy()
    current = current.merge(
        predictions[["代码", "预测下季度超额"]], on="代码", how="left", validate="one_to_one"
    )
    human = human_research.copy()
    human["human_approved"] = human["human_approved"].map(_as_bool)
    result = current.merge(human, left_on="代码", right_on="ticker", how="inner", validate="one_to_one")
    result = result.merge(narrative_summary, left_on="代码", right_on="ticker", how="left", suffixes=("", "_ai"), validate="one_to_one")
    result = result.merge(
        sector_scores[["sector", "sector_opportunity_score", "macro_regime"]],
        left_on="行业", right_on="sector", how="left", validate="many_to_one"
    )
    if result["ai_consensus"].isna().any():
        raise ValueError("Every deep-dive candidate needs narrative-agent outputs")

    weights = gaia_config["gfss_weights"]
    result["traditional_quant"] = result["TQS百分位1-5"].fillna(0.5) / 5.0
    result["fundamental"] = result["fundamental_rating_1_5"] / 5.0
    result["narrative_delta_component"] = (result["narrative_delta"].clip(-1, 1) + 1) / 2
    result["catalyst"] = result["catalyst_rating_1_5"] / 5.0
    result["management"] = result["management_rating_1_5"] / 5.0
    result["risk_component"] = (
        0.50 * result["bear_risk_rating_1_5"] / 5.0
        + 0.25 * (result["Beta"].abs() / 2.0).clip(0, 1)
        + 0.25 * (result["Vol252"] / 0.60).clip(0, 1)
    )
    result["gfss_raw"] = (
        weights["traditional_quant"] * result["traditional_quant"]
        + weights["fundamental"] * result["fundamental"]
        + weights["narrative_delta"] * result["narrative_delta_component"]
        + weights["catalyst"] * result["catalyst"]
        + weights["management"] * result["management"]
        + weights["ai_consensus"] * result["ai_consensus"]
        - weights["risk"] * result["risk_component"]
    )
    research_confidence = result["research_confidence"].map(RESEARCH_CONFIDENCE).fillna(0.50)
    data_confidence = (result["因子覆盖数"] / 5.0).clip(0, 1).fillna(0.25)
    result["confidence"] = np.sqrt(
        np.sqrt(research_confidence * result["ai_confidence"]) * data_confidence
    )
    offline = result["provider"] != "openai_compatible_llm"
    result.loc[offline, "confidence"] = result.loc[offline, "confidence"].clip(upper=0.50)
    result.loc[~result["human_approved"], "confidence"] = result.loc[~result["human_approved"], "confidence"].clip(upper=0.50)
    result["adjusted_gfss"] = result["gfss_raw"] * result["confidence"]

    aversion = _risk_aversion(year)
    beta_excess = (result["Beta"].abs() - 0.80).clip(lower=0)
    volatility_excess = (result["Vol252"] - 0.25).clip(lower=0)
    drawdown_excess = (result["最大回撤"].abs() - 0.25).clip(lower=0)
    result["goal_fit"] = (1 - aversion * (0.35 * beta_excess + 0.25 * volatility_excess)).clip(0.35, 1.0)
    result["risk_penalty"] = aversion * (
        0.10 * beta_excess + 0.05 * volatility_excess + 0.05 * drawdown_excess
    )
    result["gcc"] = result["adjusted_gfss"] * result["goal_fit"] - result["risk_penalty"]
    result["phase"] = liability_phase(year)
    result["conviction_year"] = year

    def gate(row) -> str:
        if gaia_config["uncertainty"]["require_human_approval"] and not row["human_approved"]:
            return "HUMAN_DEEP_DIVE_APPROVAL_REQUIRED"
        if row["ai_gate"] == "REVIEW_REQUIRED":
            return "AI_UNCERTAINTY_REVIEW"
        if row["confidence"] < gaia_config["uncertainty"]["min_ai_confidence"]:
            return "LOW_CONFIDENCE_REVIEW"
        if row["ads"] > gaia_config["uncertainty"]["max_ads"]:
            return "HIGH_DISAGREEMENT_REVIEW"
        return "ELIGIBLE_FOR_INVESTMENT_COMMITTEE"

    result["decision_gate"] = result.apply(gate, axis=1)
    result["eligible_for_allocation"] = result["decision_gate"] == "ELIGIBLE_FOR_INVESTMENT_COMMITTEE"
    result["research_only"] = ~result["eligible_for_allocation"]
    result["ai_signal_two_speed"] = (
        0.80 * result["slow_structural_signal"] + 0.20 * ((result["fast_information_delta"] + 1) / 2)
    )
    keep = [
        "代码", "公司", "行业", "phase", "conviction_year", "macro_regime", "sector_opportunity_score",
        "traditional_quant", "fundamental", "narrative_delta", "narrative_delta_component",
        "catalyst", "management", "ai_consensus", "ai_confidence", "ads", "risk_component",
        "gfss_raw", "confidence", "adjusted_gfss", "goal_fit", "risk_penalty", "gcc",
        "ai_signal_two_speed", "预测下季度超额", "Beta", "Vol252", "最大回撤", "ADV(百万美元)",
        "provider", "ai_gate", "gate_reasons", "human_approved", "decision_gate",
        "eligible_for_allocation", "research_only", "rationale", "source_url",
    ]
    return result[keep].sort_values("gcc", ascending=False).reset_index(drop=True)


def gfss_weight_sensitivity(conviction: pd.DataFrame, gaia_config: dict, shock: float = 0.20) -> pd.DataFrame:
    """Test rank robustness to ±20% individual GFSS weight changes, not return tuning."""
    base_weights = gaia_config["gfss_weights"]
    component_columns = {
        "traditional_quant": "traditional_quant",
        "fundamental": "fundamental",
        "narrative_delta": "narrative_delta_component",
        "catalyst": "catalyst",
        "management": "management",
        "ai_consensus": "ai_consensus",
        "risk": "risk_component",
    }
    rows = []
    for name in base_weights:
        for direction in [-1, 1]:
            weights = base_weights.copy()
            weights[name] = max(0.0, weights[name] * (1 + direction * shock))
            positive_total = sum(value for key, value in weights.items() if key != "risk")
            for key in weights:
                if key != "risk":
                    weights[key] /= positive_total
            score = pd.Series(0.0, index=conviction.index)
            for key, column in component_columns.items():
                score += (1 if key != "risk" else -1) * weights[key] * conviction[column]
            adjusted = score * conviction["confidence"]
            correlation = spearmanr(conviction["adjusted_gfss"], adjusted).statistic
            rows.append({
                "shocked_weight": name,
                "direction": direction,
                "new_weight": weights[name],
                "rank_correlation_with_base": correlation,
                "top_candidate": conviction.loc[adjusted.idxmax(), "代码"],
            })
    return pd.DataFrame(rows)
