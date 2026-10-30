import pytest

from gaia_ai.data import load_account, load_base_weights, load_positions
from gaia_ai.portfolio import generate_daily_signal


def test_portfolio_constraints_and_cash(project_root, config, latest_result):
    latest, _, _, _ = latest_result
    daily, instructions, summary = generate_daily_signal(
        predictions=latest,
        positions=load_positions(project_root / "data" / "positions_snapshot.csv"),
        account=load_account(project_root / "data" / "account_snapshot.csv"),
        base_weights=load_base_weights(project_root / "data" / "base_weights.csv"),
        config=config.portfolio,
    )
    stocks = daily[daily["ticker"] != "SPYM"]
    assert stocks["target_weight"].sum() == pytest.approx(0.4, abs=1e-10)
    assert daily["target_weight"].sum() == pytest.approx(1.0, abs=1e-10)
    assert stocks["target_weight"].min() >= config.portfolio.min_stock_weight - 1e-12
    assert stocks["target_weight"].max() <= config.portfolio.max_stock_weight + 1e-12
    assert summary["ending_cash"] >= 0
    assert summary["cash_check"] == "PASS"
    assert summary["trade_limit_check"] == "PASS"
    assert summary["remaining_trades"] == 183
    assert (instructions["executed_shares"] >= 0).all()
    assert (instructions["executed_shares"] <= instructions["theoretical_shares"]).all()


def test_small_differences_do_not_trade(project_root, config, latest_result):
    latest, _, _, _ = latest_result
    daily, _, _ = generate_daily_signal(
        latest,
        load_positions(project_root / "data" / "positions_snapshot.csv"),
        load_account(project_root / "data" / "account_snapshot.csv"),
        load_base_weights(project_root / "data" / "base_weights.csv"),
        config.portfolio,
    )
    no_trade = daily[daily["direction"] == "不交易"]
    assert (no_trade["executed_shares"] == 0).all()
    assert (no_trade["commission"] == 0).all()
