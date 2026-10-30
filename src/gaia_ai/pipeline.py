from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from .allocation import risk_constrained_allocation
from .backtest import expanding_window_backtest
from .config import ProjectConfig
from .conviction import build_conviction, gfss_weight_sensitivity
from .data import load_account, load_base_weights, load_panel, prepare_panel
from .full_reporting import write_full_workbook
from .liability import liability_map, load_client, present_value_payments, reserve_glidepath
from .model import latest_predictions
from .narrative import NarrativeEngine, build_provider, load_documents
from .reporting import write_json
from .risk import dynamic_risk_budget, risk_budget_table
from .sector import score_sectors
from .simulation import run_scenario_suite


def run_full_pipeline(
    config: ProjectConfig,
    data_dir: str | Path,
    output_dir: str | Path,
    as_of: str | pd.Timestamp,
    narrative_provider: str = "extractive",
    simulation_paths: int | None = None,
) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(as_of).normalize()

    panel = load_panel(data_dir / "gaia_quarterly_panel.csv")
    prepared = prepare_panel(panel, as_of, config.model)
    backtest, oos, model_metrics = expanding_window_backtest(prepared, config.model)
    predictions, coefficients, train_rows, score_quarter = latest_predictions(prepared, as_of, config.model)

    client = load_client(data_dir / "client_profile.json")
    account = load_account(data_dir / "account_snapshot.csv")
    base_weights = load_base_weights(data_dir / "base_weights.csv")
    market = pd.read_csv(data_dir / "market_regime.csv", encoding="utf-8-sig").iloc[0].to_dict()
    sector_input = pd.read_csv(data_dir / "sector_scorecard.csv", encoding="utf-8-sig")
    sectors, sector_metadata = score_sectors(
        sector_input, market["growth_direction"], market["inflation_direction"], config.gaia["sector_weights"]
    )
    human = pd.read_csv(data_dir / "human_research.csv", encoding="utf-8-sig")
    documents = load_documents(data_dir / "narrative_documents_demo.csv")
    provider = build_provider(narrative_provider)
    engine = NarrativeEngine(
        config.gaia["agents"], config.gaia["uncertainty"]["min_ai_confidence"], config.gaia["uncertainty"]["max_ads"]
    )
    evidence, agents, narrative_summary = engine.run(documents, provider)
    conviction = build_conviction(
        prepared, score_quarter, predictions, human, narrative_summary, sectors, int(client["base_year"]), config.gaia
    )
    sensitivity = gfss_weight_sensitivity(conviction, config.gaia)
    risk_budget = dynamic_risk_budget(
        client, int(client["base_year"]), float(account["portfolio_value"]), market, config.gaia["risk"]
    )
    positions, buckets, allocation_audit, allocation_metadata = risk_constrained_allocation(
        conviction, base_weights, risk_budget, float(account["portfolio_value"]), config.gaia
    )
    simulation_config = deepcopy(config.gaia["simulation"])
    if simulation_paths is not None:
        simulation_config["paths"] = int(simulation_paths)
    risk_schedule = risk_budget_table(
        client, market, config.gaia["risk"], float(account["portfolio_value"]),
        years=range(int(client["base_year"]), int(client["payment_end_year"]) + 1),
    )
    bucket_weight_schedule = {
        int(row.year): row.bucket_weights for row in risk_schedule.itertuples()
    }
    scenarios = run_scenario_suite(
        client, float(account["portfolio_value"]), risk_budget["bucket_weights"], simulation_config,
        bucket_weight_schedule=bucket_weight_schedule,
    )
    base_result = scenarios["results"]["base"]
    commitments = pd.DataFrame([
        {"scenario": name, **result["commitment"]}
        for name, result in scenarios["results"].items()
    ])
    liabilities = liability_map(client, int(client["base_year"]), int(client["payment_end_year"]))
    reserves = reserve_glidepath(client)

    summary = {
        "system_status": "OFFLINE_DEMO_RESEARCH_ONLY_NO_TRADE" if narrative_provider == "extractive" else "HUMAN_COMMITTEE_REVIEW_REQUIRED",
        "as_of": as_of,
        "score_quarter": score_quarter,
        "narrative_provider": provider.name,
        "narrative_provider_is_llm": provider.name == "openai_compatible_llm",
        "demo_documents": bool(documents["is_demo"].astype(str).str.lower().eq("true").all()),
        "human_approved_candidates": int(conviction["human_approved"].sum()),
        "allocation_eligible_candidates": int(conviction["eligible_for_allocation"].sum()),
        "allocation_status": allocation_metadata["status"],
        "macro_regime": sector_metadata["macro_regime"],
        "portfolio_value": float(account["portfolio_value"]),
        "reserve_target_2033": present_value_payments(client, int(client["reserve_year"])) + float(client["flexibility_buffer"]),
        "base_operating_funding_confidence": base_result["summary"]["operating_funding_confidence"],
        "base_commitment_recommended_lower": base_result["commitment"]["recommended_lower"],
        "base_commitment_recommended_upper": base_result["commitment"]["recommended_upper"],
        "base_probability_zero_facility_contribution": base_result["commitment"]["probability_zero_contribution"],
        "simulation_paths_per_scenario": base_result["summary"]["paths"],
        "simulation_bucket_schedule": base_result["summary"]["bucket_weight_schedule_status"],
        "model_final_training_rows": train_rows,
        "model_oos_ic_mean": model_metrics["ic_mean"],
        "model_oos_direction_hit_rate": model_metrics["hit_rate_quarter_average"],
        "governance": "No automatic trading. AI output cannot bypass human deep-dive approval or the investment committee.",
        "forecast_limit": "The ridge model ranks next-quarter relative returns; it does not predict daily prices with false precision.",
        "simulation_assumption_status": "Internal research assumptions, not official WGYS parameters or a market forecast.",
    }

    frames = {
        "executive_summary": pd.DataFrame({"item": list(summary), "value": [str(value) for value in summary.values()]}),
        "liability_map": liabilities,
        "regime_sectors": sectors,
        "conviction": conviction,
        "narrative_agents": agents,
        "narrative_evidence": evidence,
        "uncertainty_gate": conviction[["代码", "provider", "ai_confidence", "ads", "ai_gate", "gate_reasons", "human_approved", "decision_gate", "eligible_for_allocation"]],
        "allocation_research": positions,
        "allocation_audit": allocation_audit,
        "risk_buckets": buckets,
        "risk_schedule": risk_schedule,
        "funding_scenarios": scenarios["summary"],
        "ofc_thresholds": scenarios["thresholds"],
        "payment_probability": scenarios["payment_probability"],
        "annual_paths": scenarios["annual"],
        "commitments_2031": commitments,
        "reserve_glidepath": reserves,
        "gfss_sensitivity": sensitivity,
        "model_backtest": backtest,
        "model_coefficients": coefficients,
        "latest_predictions": predictions,
        "facility_samples_base": base_result["samples"],
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "full_system_summary.json", summary)
    write_full_workbook(output_dir / "GAIA_full_system_report.xlsx", frames, summary)
    return {"summary": summary, "frames": frames, "scenarios": scenarios, "output_dir": output_dir}
