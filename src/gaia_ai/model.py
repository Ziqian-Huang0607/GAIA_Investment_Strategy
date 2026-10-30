from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import ModelConfig


def fit_model(train: pd.DataFrame, config: ModelConfig) -> Ridge:
    if train.empty:
        raise ValueError("Cannot fit model with an empty training set")
    model = Ridge(alpha=config.alpha, positive=True, fit_intercept=True)
    model.fit(train[config.zfeatures], train["建模标签"])
    return model


def latest_predictions(
    prepared: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp]:
    """Fit all eligible history and score the latest available feature quarter."""
    as_of = pd.Timestamp(as_of).normalize()
    train = prepared[prepared["标签是否完整"]].dropna(
        subset=config.zfeatures + ["建模标签"]
    )
    model = fit_model(train, config)

    quarters = prepared.loc[prepared["季度"] <= as_of, "季度"]
    if quarters.empty:
        raise ValueError("No feature quarter exists on or before as_of")
    score_quarter = quarters.max()
    latest = prepared[prepared["季度"] == score_quarter].copy()
    latest["预测下季度超额"] = np.nan
    usable = latest.dropna(subset=config.zfeatures).index
    latest.loc[usable, "预测下季度超额"] = model.predict(
        latest.loc[usable, config.zfeatures]
    )

    latest["AI评分_全池1至5"] = np.nan
    count = len(usable)
    if count > 1:
        ranks = latest.loc[usable, "预测下季度超额"].rank(method="average")
        latest.loc[usable, "AI评分_全池1至5"] = 1 + 4 * (ranks - 1) / (count - 1)

    keep = [
        "季度", "代码", "公司", "行业", "最新申报日", "3B综合分", "3C成长",
        "3C质量", "P/E(TTM)", "E/P", "预测下季度超额", "AI评分_全池1至5",
    ]
    keep = [column for column in keep if column in latest.columns]
    latest = latest[keep].sort_values(
        "预测下季度超额", ascending=False, na_position="last"
    ).reset_index(drop=True)
    coefficients = pd.DataFrame({
        "term": list(config.features) + ["intercept"],
        "coefficient": list(model.coef_) + [model.intercept_],
    })
    return latest, coefficients, len(train), score_quarter
