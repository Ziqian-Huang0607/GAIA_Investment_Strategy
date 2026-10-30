"""Full GAIA research system with quantitative ML and human-governed decision support."""

__version__ = "2.0.0"

from .backtest import expanding_window_backtest
from .config import ModelConfig, PortfolioConfig, ProjectConfig, load_config
from .conviction import build_conviction, gfss_weight_sensitivity
from .data import load_panel, prepare_panel
from .liability import load_client, payment_schedule, present_value_payments
from .model import fit_model, latest_predictions
from .narrative import NarrativeEngine
from .pipeline import run_full_pipeline
from .portfolio import generate_daily_signal
from .risk import dynamic_risk_budget
from .sector import classify_regime, score_sectors
from .simulation import run_scenario_suite, simulate_funding

__all__ = [
    "ModelConfig", "PortfolioConfig", "ProjectConfig", "load_config",
    "load_panel", "prepare_panel", "fit_model", "latest_predictions",
    "expanding_window_backtest", "generate_daily_signal", "NarrativeEngine",
    "build_conviction", "gfss_weight_sensitivity", "dynamic_risk_budget",
    "classify_regime", "score_sectors", "load_client", "payment_schedule",
    "present_value_payments", "simulate_funding", "run_scenario_suite",
    "run_full_pipeline",
]
