from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_FITS = {
    "Recovery": {"Consumer Discretionary": 0.90, "Consumer Staples": 0.55, "Real Estate": 0.80, "Utilities": 0.45},
    "Overheat": {"Consumer Discretionary": 0.65, "Consumer Staples": 0.60, "Real Estate": 0.50, "Utilities": 0.45},
    "Stagflation": {"Consumer Discretionary": 0.30, "Consumer Staples": 0.75, "Real Estate": 0.35, "Utilities": 0.65},
    "Recession": {"Consumer Discretionary": 0.25, "Consumer Staples": 0.85, "Real Estate": 0.45, "Utilities": 0.80},
    "Disinflationary Slowdown": {"Consumer Discretionary": 0.50, "Consumer Staples": 0.70, "Real Estate": 0.75, "Utilities": 0.80},
}


def classify_regime(growth_direction: str, inflation_direction: str) -> str:
    growth = growth_direction.strip().lower()
    inflation = inflation_direction.strip().lower()
    if growth in {"rising", "improving", "up"} and inflation in {"falling", "down", "disinflation"}:
        return "Recovery"
    if growth in {"rising", "improving", "up"} and inflation in {"rising", "up"}:
        return "Overheat"
    if growth in {"slowing", "falling", "down"} and inflation in {"rising", "up"}:
        return "Stagflation"
    if growth in {"slowing", "falling", "down"} and inflation in {"falling", "down", "disinflation"}:
        return "Disinflationary Slowdown"
    if growth in {"contracting", "recession"}:
        return "Recession"
    return "Disinflationary Slowdown"


def _percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    result = values.rank(pct=True, method="average")
    if not higher_is_better:
        result = 1 - result + 1 / len(values)
    return result.fillna(0.5)


def score_sectors(
    sectors: pd.DataFrame,
    growth_direction: str,
    inflation_direction: str,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, dict]:
    required = set(weights).difference(sectors.columns)
    if required:
        raise ValueError(f"Missing sector scorecard columns: {sorted(required)}")
    result = sectors.copy()
    result["macro_regime"] = classify_regime(growth_direction, inflation_direction)
    weighted_score = pd.Series(0.0, index=result.index)
    for metric, weight in weights.items():
        if metric == "earnings_revisions":
            # Revision breadth should be signed; missing data remain neutral rather than fabricated.
            signed = result[metric].fillna(0.0)
            result[metric + "_component"] = (signed.clip(-1, 1) + 1) / 2
        else:
            result[metric + "_component"] = _percentile(result[metric], higher_is_better=True)
        weighted_score += weight * result[metric + "_component"]
    result["scorecard_sos"] = weighted_score
    result["regime_fit"] = result.apply(
        lambda row: REGIME_FITS[row["macro_regime"]].get(row["sector"], 0.50), axis=1
    )
    result["sector_opportunity_score"] = 0.80 * result["scorecard_sos"] + 0.20 * result["regime_fit"]
    result["priority_rank"] = result["sector_opportunity_score"].rank(ascending=False, method="min").astype(int)
    result["priority_sector"] = result["priority_rank"] <= min(6, len(result))
    metadata = {
        "macro_regime": str(result["macro_regime"].iloc[0]),
        "growth_direction": growth_direction,
        "inflation_direction": inflation_direction,
        "sector_count": int(len(result)),
        "coverage_warning": "Only supplied candidate sectors are ranked; this is not a complete GICS universe.",
        "earnings_revisions": "Missing values receive a neutral 0.5 component and must be replaced when available.",
    }
    return result.sort_values("sector_opportunity_score", ascending=False).reset_index(drop=True), metadata
