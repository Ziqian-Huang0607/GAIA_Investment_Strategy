from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .liability import commitment_range, payment_schedule, present_value_payments


ASSETS = ["Growth", "Stability", "FutureLiability"]


def _student_t_paths(rng: np.random.Generator, years: int, paths: int, df: int, correlation: list[list[float]]) -> np.ndarray:
    corr = np.asarray(correlation, dtype=float)
    normal = rng.multivariate_normal(np.zeros(len(ASSETS)), corr, size=(years, paths))
    chi_square = rng.chisquare(df, size=(years, paths, 1))
    return normal / np.sqrt(chi_square / df) * np.sqrt((df - 2) / df)


def _scenario_parameters(name: str, config: dict) -> dict:
    params = {
        "mean_shift": np.zeros(3),
        "vol_multiplier": np.ones(3),
        "growth_shocks": {},
        "stability_shocks": {},
        "future_liability_shocks": {},
        "reserve_mean_shift": 0.0,
    }
    scenarios = {
        "base": {},
        "bull": {"mean_shift": np.array([0.04, 0.01, 0.005]), "vol_multiplier": np.array([0.80, 0.85, 1.0])},
        "bear": {"mean_shift": np.array([-0.05, -0.01, 0.0]), "vol_multiplier": np.array([1.30, 1.25, 1.0])},
        "recession": {"mean_shift": np.array([-0.02, -0.005, 0.0]), "growth_shocks": {2027: -0.20, 2028: -0.10}, "stability_shocks": {2027: -0.04}},
        "equity_crash": {"growth_shocks": {2027: -0.35}, "stability_shocks": {2027: -0.05}},
        "rate_shock": {"growth_shocks": {2027: -0.08}, "stability_shocks": {2027: -0.12}, "future_liability_shocks": {2027: -0.08}, "reserve_mean_shift": -0.01},
        "sequence_stress": {"growth_shocks": {2027: -0.18, 2028: -0.12, 2029: -0.08}, "stability_shocks": {2027: -0.03, 2028: -0.02}},
    }
    if name not in scenarios:
        raise ValueError(f"Unknown scenario {name!r}; choose from {', '.join(scenarios)}")
    params.update(scenarios[name])
    return params


def simulate_funding(
    client: dict,
    initial_value: float,
    bucket_weights: dict[str, float],
    simulation_config: dict,
    scenario: str = "base",
    paths: int | None = None,
    seed: int | None = None,
    bucket_weight_schedule: dict[int, dict[str, float]] | None = None,
) -> dict:
    """Vectorized funding simulation without relying on sponsor money for OFC.

    Dollar returns are annual, contributions/payments occur at the beginning of the
    year, and annual returns are applied afterward. Facility commitment is an output,
    not an exogenous source of funding in the base OFC calculation.
    """
    config = deepcopy(simulation_config)
    paths = int(paths or config["paths"])
    seed = int(config["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    start_year = client["base_year"]
    final_year = int(client["payment_end_year"])
    years = list(range(start_year, final_year + 1))
    n_years = len(years)
    payment_years_list = list(range(int(client["payment_start_year"]), final_year + 1))
    payment_years = set(payment_years_list)
    annual_payment = float(client["payment_amount"])
    contributions = {int(year): float(amount) for year, amount in client["contributions"].items()}
    reserve_year = int(client["reserve_year"])

    weights = np.array([bucket_weights[name] for name in ASSETS], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Bucket weights must sum to one")
    means = np.array([config["return_assumptions"][name]["mean"] for name in ASSETS])
    vols = np.array([config["return_assumptions"][name]["vol"] for name in ASSETS])
    scenario_params = _scenario_parameters(scenario, config)
    means = means + scenario_params["mean_shift"]
    vols = vols * scenario_params["vol_multiplier"]
    z = _student_t_paths(rng, n_years, paths, int(config["student_t_df"]), config["correlation"])
    returns = means[None, None, :] + vols[None, None, :] * z
    for asset_index, key in enumerate(["growth_shocks", "stability_shocks", "future_liability_shocks"]):
        for year, shock in scenario_params[key].items():
            if year in years:
                returns[years.index(year), :, asset_index] += shock
    returns = np.clip(returns, -0.95, 1.50)
    reserve_config = config["reserve_return"]
    reserve_z = rng.standard_t(int(config["student_t_df"]), size=(n_years, paths)) * np.sqrt((int(config["student_t_df"]) - 2) / int(config["student_t_df"]))
    reserve_returns = reserve_config["mean"] + scenario_params["reserve_mean_shift"] + reserve_config["vol"] * reserve_z
    reserve_returns = np.clip(reserve_returns, -0.95, 0.50)

    portfolio = np.full(paths, float(initial_value))
    reserve = np.zeros(paths)
    facility_samples = np.zeros(paths)
    reserve_target_samples = np.zeros(paths)
    payment_success = np.zeros((paths, len(payment_years_list)), dtype=bool)
    payment_index = {year: index for index, year in enumerate(payment_years_list)}
    annual_total = np.zeros((n_years, paths))
    annual_portfolio = np.zeros((n_years, paths))
    annual_reserve = np.zeros((n_years, paths))

    for year_index, year in enumerate(years):
        if year in contributions:
            portfolio += contributions[year]
        if year == reserve_year:
            reserve_target = present_value_payments(client, year) + float(client["flexibility_buffer"])
            reserve_target_samples[:] = reserve_target
            transfer = np.minimum(portfolio, reserve_target)
            reserve += transfer
            portfolio -= transfer
            facility_samples = np.maximum(reserve_target - transfer, 0.0)
        if year in payment_years:
            from_reserve = np.minimum(reserve, annual_payment)
            reserve -= from_reserve
            remaining = annual_payment - from_reserve
            from_portfolio = np.minimum(portfolio, remaining)
            portfolio -= from_portfolio
            paid = from_reserve + from_portfolio
            payment_success[:, payment_index[year]] = paid >= annual_payment - 1e-8
        annual_portfolio[year_index] = portfolio
        annual_reserve[year_index] = reserve
        annual_total[year_index] = portfolio + reserve
        if year != final_year:
            # Exact weighted simple return; weights are rebalanced annually in this model.
            year_weights = weights if bucket_weight_schedule is None else np.array([
                bucket_weight_schedule[year][name] for name in ASSETS
            ], dtype=float)
            if not np.isclose(year_weights.sum(), 1.0):
                raise ValueError(f"Bucket weights for {year} must sum to one")
            portfolio = annual_portfolio[year_index] * (1 + returns[year_index] @ year_weights)
            reserve *= 1 + reserve_returns[year_index]

    all_funded = payment_success.all(axis=1)
    ofc = float(all_funded.mean())
    commitment = commitment_range(facility_samples)
    thresholds = pd.DataFrame([
        {
            "scenario": scenario,
            "target_confidence": threshold,
            "actual_ofc": ofc,
            "meets_threshold": ofc >= threshold,
            "facility_commitment_at_confidence": float(np.quantile(facility_samples, threshold)),
            "interpretation": "OFC excludes sponsor support; facility quantile is a contingent commitment, not base-case funding.",
        }
        for threshold in config["confidence_thresholds"]
    ])
    payment_probability = pd.DataFrame({
        "scenario": scenario,
        "year": payment_years_list,
        "funding_probability_without_sponsor": payment_success.mean(axis=0),
        "annual_payment": annual_payment,
    })
    annual = pd.DataFrame({"scenario": scenario, "year": years})
    for label, values in [("total_assets", annual_total),("portfolio", annual_portfolio),("reserve", annual_reserve)]:
        annual[f"{label}_p10"] = np.quantile(values, 0.10, axis=1)
        annual[f"{label}_p50"] = np.quantile(values, 0.50, axis=1)
        annual[f"{label}_p90"] = np.quantile(values, 0.90, axis=1)
    summary = {
        "scenario": scenario,
        "paths": paths,
        "seed": seed,
        "return_distribution": f"Correlated Student-t, df={config['student_t_df']}",
        "initial_value": initial_value,
        "bucket_weights": bucket_weights,
        "bucket_weight_schedule_status": "Dynamic liability glidepath" if bucket_weight_schedule is not None else "Fixed initial weights",
        "operating_funding_confidence": ofc,
        "all_ten_payments_funded_probability": ofc,
        "expected_number_of_payments_funded": float(payment_success.sum(axis=1).mean()),
        "reserve_target_2033": float(reserve_target_samples[0]),
        "reserve_shortfall_probability": float((facility_samples > 0).mean()),
        "median_total_assets_end_2042": float(np.median(annual_total[-1])),
        "p10_total_assets_end_2042": float(np.quantile(annual_total[-1], 0.10)),
        "probability_any_payment_failure": float(1 - ofc),
        "assumption_status": "Internal research assumptions, not official WGYS parameters or a market forecast.",
        "sponsor_support_in_ofc": False,
        "payment_timing": "Beginning of year; annual returns applied afterward.",
        "reserve_rule": "Reserve funded before first 2033 payment, including flexibility buffer.",
    }
    summary.update({f"commitment_{key}": value for key, value in commitment.items()})
    samples = pd.DataFrame({
        "scenario": scenario,
        "path": np.arange(paths),
        "facility_commitment_2033": facility_samples,
        "all_payments_funded_without_sponsor": all_funded,
        "payments_funded": payment_success.sum(axis=1),
        "end_2042_total_assets": annual_total[-1],
    })
    return {
        "summary": summary,
        "commitment": commitment,
        "thresholds": thresholds,
        "payment_probability": payment_probability,
        "annual": annual,
        "samples": samples,
    }


def run_scenario_suite(
    client: dict,
    initial_value: float,
    bucket_weights: dict[str, float],
    simulation_config: dict,
    bucket_weight_schedule: dict[int, dict[str, float]] | None = None,
) -> dict:
    names = simulation_config.get("scenarios", ["base", "bull", "bear", "recession", "equity_crash", "rate_shock", "sequence_stress"])
    results = {
        name: simulate_funding(
            client, initial_value, bucket_weights, simulation_config,
            scenario=name, bucket_weight_schedule=bucket_weight_schedule,
        )
        for name in names
    }
    return {
        "results": results,
        "summary": pd.DataFrame([result["summary"] for result in results.values()]),
        "thresholds": pd.concat([result["thresholds"] for result in results.values()], ignore_index=True),
        "payment_probability": pd.concat([result["payment_probability"] for result in results.values()], ignore_index=True),
        "annual": pd.concat([result["annual"] for result in results.values()], ignore_index=True),
    }
