from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .gaia_defaults import DEFAULT_GAIA


@dataclass(frozen=True)
class ModelConfig:
    alpha: float = 10.0
    stale_days: int = 400
    min_train_quarters: int = 3
    min_test_names: int = 10
    features: Sequence[str] = ("3B综合分", "3C成长", "3C质量", "E/P")
    label: str = "下季度超额"

    @property
    def zfeatures(self) -> list[str]:
        return [f"{name}_z" for name in self.features]


@dataclass(frozen=True)
class PortfolioConfig:
    stock_sleeve_weight: float = 0.4
    max_tilt: float = 0.35
    min_stock_weight: float = 0.02
    max_stock_weight: float = 0.06
    trade_threshold_pp: float = 0.8
    commission_per_trade: float = 25.0
    trade_limit: int = 200

    @property
    def etf_weight(self) -> float:
        return 1.0 - self.stock_sleeve_weight


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    gaia: dict = field(default_factory=lambda: deepcopy(DEFAULT_GAIA))


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    model = ModelConfig(**raw.get("model", {}))
    portfolio = PortfolioConfig(**raw.get("portfolio", {}))
    if model.alpha < 0:
        raise ValueError("model.alpha must be non-negative")
    if model.min_train_quarters < 1:
        raise ValueError("model.min_train_quarters must be at least 1")
    if not 0 < portfolio.stock_sleeve_weight < 1:
        raise ValueError("portfolio.stock_sleeve_weight must be between 0 and 1")
    if not 0 <= portfolio.min_stock_weight <= portfolio.max_stock_weight:
        raise ValueError("invalid stock weight bounds")
    gaia = _deep_merge(DEFAULT_GAIA, raw.get("gaia", {}))
    return ProjectConfig(model=model, portfolio=portfolio, gaia=gaia)
