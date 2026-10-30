from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from .backtest import expanding_window_backtest
from .config import load_config
from .data import (
    load_account,
    load_base_weights,
    load_panel,
    load_positions,
    prepare_panel,
)
from .model import latest_predictions
from .portfolio import generate_daily_signal
from .pipeline import run_full_pipeline
from .reporting import write_daily_workbook, write_json

ROOT = Path(__file__).resolve().parents[2]


def project_path(relative: str) -> str:
    return str(ROOT / relative)


def run_training(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else pd.Timestamp.today().normalize()

    panel = load_panel(args.data)
    prepared = prepare_panel(panel, as_of, config.model)
    backtest, oos, metrics = expanding_window_backtest(prepared, config.model)
    latest, coefficients, train_rows, score_quarter = latest_predictions(
        prepared, as_of, config.model
    )

    latest.to_csv(output_dir / "latest_predictions.csv", index=False, encoding="utf-8-sig")
    backtest.to_csv(output_dir / "backtest_by_quarter.csv", index=False, encoding="utf-8-sig")
    oos.to_csv(output_dir / "oos_predictions.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False, encoding="utf-8-sig")
    prepared.to_csv(output_dir / "cleaned_quarterly_panel.csv", index=False, encoding="utf-8-sig")
    metadata = {
        **metrics,
        "as_of": as_of,
        "score_quarter": score_quarter,
        "final_training_rows": train_rows,
        "stale_rows_excluded": int(prepared["财报是否陈旧"].sum()),
        "incomplete_labels_masked": int((~prepared["标签是否完整"]).sum()),
        "alpha": config.model.alpha,
        "features": list(config.model.features),
    }
    write_json(output_dir / "training_summary.json", metadata)
    return {
        "latest": latest,
        "coefficients": coefficients,
        "backtest": backtest,
        "metrics": metadata,
        "output_dir": output_dir,
    }


def run_daily(args: argparse.Namespace, training: dict | None = None) -> dict:
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if training is None:
        latest = pd.read_csv(args.predictions, encoding="utf-8-sig")
    else:
        latest = training["latest"]

    positions = load_positions(args.positions)
    account = load_account(args.account)
    base_weights = load_base_weights(args.base_weights)
    daily, instructions, summary = generate_daily_signal(
        latest, positions, account, base_weights, config.portfolio
    )
    daily.to_csv(output_dir / "daily_positions.csv", index=False, encoding="utf-8-sig")
    instructions.to_csv(output_dir / "trade_review.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "daily_summary.json", summary)
    write_daily_workbook(
        output_dir / "GAIA_daily_signal.xlsx", daily, instructions, summary, latest
    )
    return {"daily": daily, "instructions": instructions, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaia-ai",
        description="GAIA quarterly ranking, walk-forward backtest, and daily portfolio signals",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def shared(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--config", default=project_path("config/default.json"))
        command_parser.add_argument("--output-dir", default=project_path("outputs"))

    train = subparsers.add_parser("train", help="Train, backtest, and score the latest quarter")
    train.add_argument("--data", default=project_path("data/gaia_quarterly_panel.csv"))
    train.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today")
    shared(train)

    daily = subparsers.add_parser("daily", help="Generate daily weights and cash-constrained trade flags")
    daily.add_argument("--predictions", default=project_path("outputs/latest_predictions.csv"))
    daily.add_argument("--positions", default=project_path("data/positions_snapshot.csv"))
    daily.add_argument("--account", default=project_path("data/account_snapshot.csv"))
    daily.add_argument("--base-weights", default=project_path("data/base_weights.csv"))
    shared(daily)

    run = subparsers.add_parser("run", help="Run the complete train-to-daily pipeline")
    run.add_argument("--data", default=project_path("data/gaia_quarterly_panel.csv"))
    run.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today")
    run.add_argument("--positions", default=project_path("data/positions_snapshot.csv"))
    run.add_argument("--account", default=project_path("data/account_snapshot.csv"))
    run.add_argument("--base-weights", default=project_path("data/base_weights.csv"))
    shared(run)

    full = subparsers.add_parser("full", help="Run all ten GAIA modules and produce the full audit report")
    full.add_argument("--data-dir", default=project_path("data"))
    full.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today")
    full.add_argument("--narrative-provider", choices=["extractive", "openai"], default="extractive")
    full.add_argument("--simulation-paths", type=int, default=None, help="Override paths per scenario")
    shared(full)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        result = run_training(args)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2, default=str))
    elif args.command == "daily":
        result = run_daily(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str))
    elif args.command == "run":
        training = run_training(args)
        result = run_daily(args, training=training)
        print("Training summary:")
        print(json.dumps(training["metrics"], ensure_ascii=False, indent=2, default=str))
        print("Daily summary:")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str))
        print(f"Outputs written to: {Path(args.output_dir).resolve()}")
    elif args.command == "full":
        as_of = args.as_of or pd.Timestamp.today().normalize()
        result = run_full_pipeline(
            load_config(args.config), args.data_dir, args.output_dir, as_of,
            narrative_provider=args.narrative_provider,
            simulation_paths=args.simulation_paths,
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str))
        print(f"Full GAIA outputs written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
