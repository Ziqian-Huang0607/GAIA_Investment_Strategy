from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_client(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        client = json.load(handle)
    required = {"contributions", "payment_amount", "payment_start_year", "payment_end_year", "reserve_year"}
    missing = required.difference(client)
    if missing:
        raise ValueError(f"Missing client fields: {sorted(missing)}")
    if client["payment_end_year"] < client["payment_start_year"]:
        raise ValueError("Invalid payment years")
    return client


def payment_schedule(client: dict) -> pd.DataFrame:
    years = range(int(client["payment_start_year"]), int(client["payment_end_year"]) + 1)
    return pd.DataFrame({
        "year": list(years),
        "payment": float(client["payment_amount"]),
        "timing": client.get("payment_timing", "beginning_of_year"),
    })


def present_value_payments(client: dict, at_year: int, discount_rate: float | None = None) -> float:
    """PV of remaining beginning-of-year payments at the start of at_year."""
    rate = float(client.get("reserve_discount_rate", 0.02) if discount_rate is None else discount_rate)
    schedule = payment_schedule(client)
    remaining = schedule[schedule["year"] >= at_year]
    return float(sum(
        row.payment / ((1 + rate) ** (int(row.year) - at_year))
        for row in remaining.itertuples()
    ))


def liability_map(client: dict, start_year: int = 2026, end_year: int = 2042) -> pd.DataFrame:
    rows = []
    for year in range(start_year, end_year + 1):
        required = present_value_payments(client, year)
        rows.append({
            "year": year,
            "year_number": year - int(client.get("base_year", 2026)),
            "contribution": float(client["contributions"].get(str(year), 0.0)),
            "operating_payment": float(client["payment_amount"]) if client["payment_start_year"] <= year <= client["payment_end_year"] else 0.0,
            "required_funding_path": required,
            "years_to_first_payment": max(int(client["payment_start_year"]) - year, 0),
            "phase": liability_phase(year),
        })
    return pd.DataFrame(rows)


def funding_metrics(client: dict, year: int, portfolio_value: float) -> dict[str, float | int | str]:
    required = present_value_payments(client, year)
    ratio = portfolio_value / required if required > 0 else np.inf
    return {
        "year": year,
        "phase": liability_phase(year),
        "portfolio_value": float(portfolio_value),
        "required_funding_path": float(required),
        "funding_ratio": float(ratio),
        "funding_surplus": float(portfolio_value - required),
    }


def liability_phase(year: int) -> str:
    if year <= 2029:
        return "Growth"
    if year <= 2031:
        return "Transition"
    if year == 2032:
        return "Protection"
    return "Liability Segmentation"


def reserve_glidepath(client: dict) -> pd.DataFrame:
    rows = []
    for year in range(int(client["reserve_year"]), int(client["payment_end_year"]) + 1):
        elapsed = year - int(client["reserve_year"])
        liability_weight = min(1.0, 0.80 + 0.02 * elapsed)
        rows.append({
            "year": year,
            "future_liability_weight": liability_weight,
            "stability_weight": 1 - liability_weight,
            "beginning_payment": float(client["payment_amount"]),
            "rule": "Pay operating commitment first; shorten duration and increase liquidity as payments approach.",
        })
    return pd.DataFrame(rows)


def commitment_range(
    facility_samples: np.ndarray,
    lower_percentile: float = 10,
    upper_percentile: float = 50,
) -> dict[str, float]:
    samples = np.asarray(facility_samples, dtype=float)
    if samples.size == 0:
        raise ValueError("Empty facility contribution sample")
    percentiles = {f"p{str(p).replace('.', '_')}": float(np.percentile(samples, p)) for p in [5, 10, 25, 50, 75, 90, 95, 97.5, 99]}
    lower = float(np.percentile(samples, lower_percentile))
    upper = float(np.percentile(samples, upper_percentile))
    return {
        **percentiles,
        "recommended_lower": lower,
        "recommended_upper": upper,
        "range_coverage": float(np.mean((samples >= lower) & (samples <= upper))),
        "probability_zero_contribution": float(np.mean(samples <= 1e-8)),
        "expected_contribution": float(samples.mean()),
        "expected_contribution_when_positive": float(samples[samples > 1e-8].mean()) if np.any(samples > 1e-8) else 0.0,
        "maximum_contribution": float(samples.max()),
    }
