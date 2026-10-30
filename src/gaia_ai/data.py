from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import ModelConfig


PANEL_REQUIRED = {
    "季度", "代码", "最新申报日", "P/E(TTM)", "3B综合分", "3C成长", "3C质量", "下季度超额"
}


def load_panel(path: str | Path) -> pd.DataFrame:
    """Load the quarterly panel from CSV or the original Excel workbook."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        panel = pd.read_csv(path, encoding="utf-8-sig")
    elif path.suffix.lower() in {".xlsx", ".xlsm"}:
        panel = pd.read_excel(path, sheet_name="05_季度指标面板", header=2)
    else:
        raise ValueError("Panel must be .csv, .xlsx, or .xlsm")
    missing = PANEL_REQUIRED.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing panel columns: {sorted(missing)}")
    return panel


def prepare_panel(panel: pd.DataFrame, as_of: str | pd.Timestamp, config: ModelConfig) -> pd.DataFrame:
    """Create model features while enforcing point-in-time and data-quality rules."""
    d = panel.copy()
    as_of = pd.Timestamp(as_of).normalize()
    d["季度"] = pd.to_datetime(d["季度"], errors="coerce")
    d["最新申报日"] = pd.to_datetime(d["最新申报日"], errors="coerce")
    if d[["季度", "代码"]].isna().any().any():
        raise ValueError("Quarter or ticker contains invalid/missing values")
    if d.duplicated(["季度", "代码"]).any():
        raise ValueError("Duplicate quarter-ticker keys found")

    d["标签季度末"] = d["季度"] + pd.offsets.QuarterEnd(1)
    d["标签是否完整"] = d["标签季度末"] <= as_of
    d["建模标签"] = pd.to_numeric(d[config.label], errors="coerce").where(
        d["标签是否完整"], np.nan
    )

    pe = pd.to_numeric(d["P/E(TTM)"], errors="coerce")
    d["E/P"] = np.where(pe > 0, 1.0 / pe, np.nan)
    d["财报距季度末天数"] = (d["季度"] - d["最新申报日"]).dt.days
    d["财报是否陈旧"] = d["财报距季度末天数"] > config.stale_days
    for column in ["3C成长", "3C质量", "E/P"]:
        d.loc[d["财报是否陈旧"], column] = np.nan

    for column in config.features:
        d[column] = pd.to_numeric(d[column], errors="coerce")
        d[f"{column}_z"] = d.groupby("季度")[column].transform(
            lambda values: (values - values.mean()) / values.std(ddof=0)
        )
    return d


def load_account(path: str | Path) -> dict[str, float | str]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if not {"field", "value"}.issubset(frame.columns):
        raise ValueError("Account CSV needs field and value columns")
    values = frame.set_index("field")["value"].to_dict()
    required = {"portfolio_value", "total_return", "benchmark_return", "cash", "trades_used"}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"Missing account fields: {sorted(missing)}")
    for key in required:
        values[key] = float(values[key])
    if "as_of" in frame.columns and not frame["as_of"].dropna().empty:
        values["as_of"] = str(frame["as_of"].dropna().iloc[0])
    return values


def load_positions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"股票", "数量", "最新价"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing position columns: {sorted(missing)}")
    if frame["股票"].duplicated().any():
        raise ValueError("Duplicate tickers in positions CSV")
    frame["数量"] = pd.to_numeric(frame["数量"], errors="raise")
    frame["最新价"] = pd.to_numeric(frame["最新价"], errors="raise")
    if (frame[["数量", "最新价"]] < 0).any().any():
        raise ValueError("Position quantities and prices must be non-negative")
    return frame


def load_base_weights(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if not {"ticker", "base_total_weight"}.issubset(frame.columns):
        raise ValueError("Base weight CSV needs ticker and base_total_weight")
    frame["base_total_weight"] = pd.to_numeric(frame["base_total_weight"], errors="raise")
    if frame["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers in base weights CSV")
    return frame
