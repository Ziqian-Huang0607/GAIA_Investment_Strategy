from __future__ import annotations

import numpy as np
import pandas as pd


def _capped_weights(scores: pd.Series, sectors: pd.Series, target: float, min_weight: float, max_weight: float, sector_cap: float) -> pd.Series:
    if target <= 0 or scores.empty:
        return pd.Series(0.0, index=scores.index)
    positive = scores.clip(lower=0)
    if positive.sum() <= 0:
        positive = pd.Series(1.0, index=scores.index)
    weights = positive / positive.sum() * target
    for _ in range(200):
        weights = weights.clip(lower=min_weight, upper=max_weight)
        for sector in sectors.unique():
            mask = sectors == sector
            total = weights[mask].sum()
            if total > sector_cap and total > 0:
                weights.loc[mask] *= sector_cap / total
        difference = target - weights.sum()
        if abs(difference) < 1e-10:
            break
        if difference > 0:
            sector_remaining = sectors.map(weights.groupby(sectors).sum()).rsub(sector_cap).clip(lower=0)
            capacity = (max_weight - weights).clip(lower=0)
            capacity = np.minimum(capacity, sector_remaining)
            if capacity.sum() <= 1e-12:
                break
            weights += difference * capacity / capacity.sum()
        else:
            capacity = (weights - min_weight).clip(lower=0)
            if capacity.sum() <= 1e-12:
                break
            weights += difference * capacity / capacity.sum()
    return weights


def risk_constrained_allocation(
    conviction: pd.DataFrame,
    base_weights: pd.DataFrame,
    risk_budget: dict,
    portfolio_value: float,
    gaia_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    constraints = gaia_config["allocation"]
    stock_sleeve = constraints["stock_sleeve"]
    candidates = conviction.copy()
    candidates["score_for_allocation"] = (candidates["gcc"] - candidates["gcc"].min() + 0.01).clip(lower=0.001)
    candidates["research_weight"] = _capped_weights(
        candidates["score_for_allocation"], candidates["行业"], stock_sleeve,
        constraints["min_stock_weight"], constraints["max_stock_weight"], constraints["sector_cap_within_stock_sleeve"] * stock_sleeve,
    )
    eligible = candidates["eligible_for_allocation"]
    candidates["approved_target_weight"] = 0.0
    if eligible.any():
        candidates.loc[eligible, "approved_target_weight"] = _capped_weights(
            candidates.loc[eligible, "score_for_allocation"], candidates.loc[eligible, "行业"], stock_sleeve,
            constraints["min_stock_weight"], constraints["max_stock_weight"], constraints["sector_cap_within_stock_sleeve"] * stock_sleeve,
        )
    base = base_weights.set_index("ticker")["base_total_weight"].to_dict()
    candidates["base_weight"] = candidates["代码"].map(base).fillna(0.0)
    candidates["research_change"] = candidates["research_weight"] - candidates["base_weight"]
    candidates["approved_change"] = candidates["approved_target_weight"] - candidates["base_weight"]
    candidates["research_dollars"] = candidates["research_weight"] * portfolio_value
    candidates["approved_dollars"] = candidates["approved_target_weight"] * portfolio_value
    candidates["action"] = np.where(
        candidates["research_only"], "RESEARCH_ONLY__NO_TRADE",
        np.where(candidates["approved_change"].abs() < 0.008, "HOLD", "COMMITTEE_REVIEW_REQUIRED"),
    )
    candidates["redundancy_proxy"] = candidates["行业"].map(candidates.groupby("行业")["代码"].transform("count"))
    candidates["redundancy_note"] = "Same-sector concentration proxy; daily-return correlation requires a separate return history."

    spym = pd.DataFrame([{
        "代码": "SPYM", "公司": "SPDR Portfolio S&P 500 ETF", "行业": "Broad Market ETF",
        "gcc": np.nan, "eligible_for_allocation": True, "research_only": False,
        "base_weight": base.get("SPYM", 1 - stock_sleeve),
        "research_weight": 1 - stock_sleeve,
        "approved_target_weight": 1 - stock_sleeve,
        "research_change": 0.0, "approved_change": 0.0,
        "research_dollars": (1 - stock_sleeve) * portfolio_value,
        "approved_dollars": (1 - stock_sleeve) * portfolio_value,
        "action": "STRATEGIC_CORE", "redundancy_proxy": 1,
        "redundancy_note": "Strategic broad-market core.",
    }])
    positions = pd.concat([spym, candidates], ignore_index=True, sort=False)

    bucket_weights = risk_budget["bucket_weights"]
    buckets = pd.DataFrame([
        {"bucket": key, "target_weight": value, "target_dollars": value * portfolio_value,
         "purpose": purpose}
        for (key, value), purpose in zip(bucket_weights.items(), [
            "Long-term growth and equity risk budget",
            "Capital stability and drawdown control",
            "Future liability pre-funding and reserve glidepath",
        ])
    ])

    stock_rows = positions[positions["代码"] != "SPYM"]
    approved_sum = stock_rows["approved_target_weight"].sum()
    sector_weights = stock_rows.groupby("行业")["approved_target_weight"].sum()
    research_stock_sum = stock_rows["research_weight"].sum()
    audit_rows = [
        {"check": "Research stock sleeve", "passed": bool(abs(research_stock_sum - stock_sleeve) < 1e-6), "detail": f"Research stock weight {research_stock_sum:.4f}; target {stock_sleeve:.4f}."},
        {"check": "Human approval before allocation", "passed": bool((stock_rows.loc[stock_rows["research_only"], "approved_target_weight"] == 0).all()), "detail": "Unapproved candidates receive zero approved target weight."},
        {"check": "Strategic SPYM weight", "passed": bool(abs(positions.loc[positions["代码"] == "SPYM", "approved_target_weight"].iloc[0] - (1 - stock_sleeve)) < 1e-9), "detail": f"Target {1-stock_sleeve:.1%}."},
        {"check": "Approved stock sleeve", "passed": bool(approved_sum == 0 or abs(approved_sum - stock_sleeve) < 1e-6), "detail": f"Approved stock weight {approved_sum:.4f}; zero means the candidate set is research-only."},
        {"check": "Individual stock maximum", "passed": bool(stock_rows["approved_target_weight"].max() <= constraints["max_stock_weight"] + 1e-9), "detail": f"Maximum {constraints['max_stock_weight']:.1%} of total portfolio."},
        {"check": "Sector redundancy cap", "passed": bool(sector_weights.empty or sector_weights.max() <= constraints["sector_cap_within_stock_sleeve"] * stock_sleeve + 1e-6), "detail": "Sector cap within the stock sleeve."},
        {"check": "Risk buckets sum", "passed": bool(abs(sum(bucket_weights.values()) - 1) < 1e-9), "detail": "Growth + Stability + Future Liability = 100%."},
    ]
    audit = pd.DataFrame(audit_rows)
    status = "RESEARCH_ONLY_NO_TRADE" if not eligible.any() else "COMMITTEE_APPROVAL_STILL_REQUIRED"
    metadata = {
        "status": status,
        "portfolio_value": portfolio_value,
        "research_stock_sleeve": stock_sleeve,
        "approved_stock_weight": float(approved_sum),
        "unallocated_due_to_gate": float(stock_sleeve - approved_sum),
        "not_an_order_ticket": True,
        "correlation_note": "Same-sector grouping is a transparent redundancy proxy, not a claim of estimated daily-return correlation.",
    }
    return positions, buckets, audit, metadata
