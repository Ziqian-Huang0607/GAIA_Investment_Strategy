from __future__ import annotations

import numpy as np
import pandas as pd

from .liability import funding_metrics, liability_phase


def _lookup_schedule(schedule: dict, year: int, after_last: float) -> float:
    points = sorted((int(key), float(value)) for key, value in schedule.items())
    if year <= points[0][0]:
        return points[0][1]
    if year >= points[-1][0]:
        return after_last
    for (left_year, left), (right_year, right) in zip(points, points[1:]):
        if left_year <= year <= right_year:
            fraction = (year - left_year) / (right_year - left_year)
            return left + fraction * (right - left)
    return after_last


def dynamic_risk_budget(
    client: dict,
    year: int,
    portfolio_value: float,
    market: dict,
    risk_config: dict,
) -> dict[str, object]:
    funding = funding_metrics(client, year, portfolio_value)
    time_multiplier = _lookup_schedule(risk_config["time_multipliers"], year, 0.05)
    liability_target = _lookup_schedule(risk_config["liability_targets"], year, 1.0)
    ratio = float(funding["funding_ratio"])

    # Ahead-of-path outcomes harvest risk; being behind never automatically increases risk.
    funding_multiplier = 1.0 if ratio <= 1.10 else max(0.70, 1.0 - 0.35 * (ratio - 1.10))
    volatility_percentile = float(market.get("spy_vol_percentile", 0.50))
    watch = float(risk_config["high_volatility_percentile"])
    extreme = float(risk_config["extreme_volatility_percentile"])
    if volatility_percentile <= watch:
        uncertainty_multiplier = 1.0
    elif volatility_percentile >= extreme:
        uncertainty_multiplier = 0.70
    else:
        uncertainty_multiplier = 1.0 - 0.30 * (volatility_percentile - watch) / (extreme - watch)

    drawdown = float(market.get("portfolio_drawdown", 0.0))
    if drawdown <= float(risk_config["drawdown_stress"]):
        drawdown_multiplier = 0.70
    elif drawdown <= float(risk_config["drawdown_watch"]):
        drawdown_multiplier = 0.85
    else:
        drawdown_multiplier = 1.0

    raw_growth = (
        float(risk_config["base_growth_weight"]) * time_multiplier * funding_multiplier
        * uncertainty_multiplier * drawdown_multiplier
    )
    growth = float(np.clip(
        raw_growth,
        risk_config["min_growth_weight"],
        risk_config["max_growth_weight"],
    ))
    growth = min(growth, 1.0 - liability_target)
    stability = 1.0 - liability_target - growth
    return {
        **funding,
        "phase": liability_phase(year),
        "time_multiplier": time_multiplier,
        "funding_multiplier": funding_multiplier,
        "uncertainty_multiplier": uncertainty_multiplier,
        "drawdown_multiplier": drawdown_multiplier,
        "growth_weight": growth,
        "stability_weight": stability,
        "future_liability_weight": liability_target,
        "bucket_weights": {
            "Growth": growth,
            "Stability": stability,
            "FutureLiability": liability_target,
        },
        "risk_budget_note": "Internal rules, not official client loss limits. Being underfunded does not authorize extra risk-taking.",
    }


def risk_budget_table(
    client: dict,
    market: dict,
    risk_config: dict,
    initial_portfolio: float,
    years: range = range(2027, 2034),
) -> pd.DataFrame:
    rows = []
    value = initial_portfolio
    for year in years:
        value += float(client["contributions"].get(str(year), 0.0))
        row = dynamic_risk_budget(client, year, value, market, risk_config)
        rows.append(row)
        # Illustrative deterministic 5% roll-forward for reviewing the rule, not a forecast.
        value *= 1.05
    return pd.DataFrame(rows)
