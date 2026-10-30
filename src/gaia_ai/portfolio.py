from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PortfolioConfig


def clip_normalize(
    weights: pd.Series,
    total: float,
    lower: float,
    upper: float,
    iterations: int = 100,
) -> pd.Series:
    """Normalize a weight vector while enforcing lower/upper bounds."""
    result = weights.astype(float).copy()
    for _ in range(iterations):
        result = result.clip(lower, upper)
        free = (result > lower + 1e-12) & (result < upper - 1e-12)
        difference = total - result.sum()
        if abs(difference) < 1e-12 or free.sum() == 0:
            break
        result.loc[free] += difference / free.sum()
    if abs(result.sum() - total) > 1e-8:
        raise ValueError("Could not normalize stock targets within the configured bounds")
    return result


def generate_daily_signal(
    predictions: pd.DataFrame,
    positions: pd.DataFrame,
    account: dict[str, float | str],
    base_weights: pd.DataFrame,
    config: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    """Map model ranks to bounded targets and cash-constrained whole-share flags."""
    daily = positions.copy().rename(columns={"股票": "ticker", "数量": "quantity", "最新价": "price"})
    prediction_map = predictions.set_index("代码")["预测下季度超额"].to_dict()
    base_map = base_weights.set_index("ticker")["base_total_weight"].to_dict()
    missing_base = set(daily["ticker"]).difference(base_map)
    if missing_base:
        raise ValueError(f"Missing base weights for: {sorted(missing_base)}")

    daily["predicted_excess"] = daily["ticker"].map(prediction_map)
    daily["base_weight"] = daily["ticker"].map(base_map)
    daily["ai_score"] = np.nan
    daily["tilt"] = 0.0

    stock_mask = daily["ticker"] != "SPYM"
    valid_mask = stock_mask & daily["predicted_excess"].notna()
    valid_count = int(valid_mask.sum())
    if valid_count > 1:
        ranks = daily.loc[valid_mask, "predicted_excess"].rank(method="average")
        daily.loc[valid_mask, "ai_score"] = 1 + 4 * (ranks - 1) / (valid_count - 1)
        daily.loc[valid_mask, "tilt"] = (
            (daily.loc[valid_mask, "ai_score"] - 3) / 2 * config.max_tilt
        )

    daily["raw_target_weight"] = daily["base_weight"] * (1 + daily["tilt"])
    daily.loc[stock_mask, "target_weight"] = clip_normalize(
        daily.loc[stock_mask, "raw_target_weight"],
        total=config.stock_sleeve_weight,
        lower=config.min_stock_weight,
        upper=config.max_stock_weight,
    )
    daily.loc[~stock_mask, "target_weight"] = config.etf_weight

    portfolio_value = float(account["portfolio_value"])
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    daily["market_value"] = daily["quantity"] * daily["price"]
    daily["current_weight"] = daily["market_value"] / portfolio_value
    daily["difference_pp"] = (daily["target_weight"] - daily["current_weight"]) * 100
    daily["theoretical_shares"] = np.floor(
        abs((daily["target_weight"] - daily["current_weight"]) * portfolio_value)
        / daily["price"].replace(0, np.nan)
    ).fillna(0).astype(int)

    conditions = [
        (daily["difference_pp"].abs() < config.trade_threshold_pp)
        | (daily["theoretical_shares"] == 0),
        daily["target_weight"] > daily["current_weight"],
    ]
    daily["direction"] = np.select(conditions, ["不交易", "买入"], default="卖出")
    daily["theoretical_amount"] = np.where(
        daily["direction"] == "不交易",
        0.0,
        daily["theoretical_shares"] * daily["price"],
    )

    theoretical_sells = daily.loc[daily["direction"] == "卖出", "theoretical_amount"].sum()
    theoretical_buys = daily.loc[daily["direction"] == "买入", "theoretical_amount"].sum()
    theoretical_trade_count = int((daily["direction"] != "不交易").sum())
    theoretical_commissions = theoretical_trade_count * config.commission_per_trade
    available_for_buys = float(account["cash"]) + theoretical_sells - theoretical_commissions
    buy_scale = 1.0 if theoretical_buys == 0 else max(0.0, min(1.0, available_for_buys / theoretical_buys))

    daily["executed_shares"] = 0
    sell_mask = daily["direction"] == "卖出"
    buy_mask = daily["direction"] == "买入"
    daily.loc[sell_mask, "executed_shares"] = daily.loc[sell_mask, "theoretical_shares"]
    daily.loc[buy_mask, "executed_shares"] = np.floor(
        daily.loc[buy_mask, "theoretical_shares"] * buy_scale
    ).astype(int)
    daily["executed_amount"] = daily["executed_shares"] * daily["price"]
    daily["commission"] = np.where(
        daily["executed_shares"] > 0, config.commission_per_trade, 0.0
    )
    daily["review_status"] = np.where(
        daily["executed_shares"] > 0, "待投委会复核", "无需交易"
    )

    actual_sells = daily.loc[sell_mask, "executed_amount"].sum()
    actual_buys = daily.loc[buy_mask, "executed_amount"].sum()
    actual_commissions = daily["commission"].sum()
    ending_cash = float(account["cash"]) + actual_sells - actual_buys - actual_commissions
    actual_trade_count = int((daily["executed_shares"] > 0).sum())
    remaining_trades = int(config.trade_limit - float(account["trades_used"]) - actual_trade_count)

    summary: dict[str, float | str] = {
        "account_as_of": str(account.get("as_of", "")),
        "theoretical_sell_proceeds": float(theoretical_sells),
        "theoretical_buy_requirement": float(theoretical_buys),
        "buy_scale": float(buy_scale),
        "actual_sell_proceeds": float(actual_sells),
        "actual_buy_cost": float(actual_buys),
        "actual_commissions": float(actual_commissions),
        "ending_cash": float(ending_cash),
        "actual_trade_count": actual_trade_count,
        "remaining_trades": remaining_trades,
        "cash_check": "PASS" if ending_cash >= -1e-8 else "FAIL",
        "trade_limit_check": "PASS" if remaining_trades >= 0 else "FAIL",
        "stock_target_sum": float(daily.loc[stock_mask, "target_weight"].sum()),
        "total_target_sum": float(daily["target_weight"].sum()),
        "note": "Research signals only; investment committee review is required before trading.",
    }

    instruction_columns = [
        "ticker", "direction", "theoretical_shares", "executed_shares", "price",
        "executed_amount", "commission", "current_weight", "target_weight",
        "difference_pp", "review_status",
    ]
    instructions = daily[instruction_columns].copy()
    return daily, instructions, summary
