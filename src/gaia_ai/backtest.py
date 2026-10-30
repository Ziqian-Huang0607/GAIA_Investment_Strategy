from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import ModelConfig
from .model import fit_model


def expanding_window_backtest(
    prepared: pd.DataFrame,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Walk forward quarter by quarter; no test-quarter data enters its own fit."""
    quarters = sorted(prepared.loc[prepared["标签是否完整"], "季度"].unique())
    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for index, quarter in enumerate(quarters):
        if index < config.min_train_quarters:
            continue
        train = prepared[
            (prepared["季度"] < quarter) & prepared["标签是否完整"]
        ].dropna(subset=config.zfeatures + ["建模标签"])
        test = prepared[
            (prepared["季度"] == quarter) & prepared["标签是否完整"]
        ].dropna(subset=config.zfeatures + ["建模标签"]).copy()
        if train.empty or len(test) < config.min_test_names:
            continue

        model = fit_model(train, config)
        test["预测超额"] = model.predict(test[config.zfeatures])
        ic = spearmanr(test["预测超额"], test["建模标签"]).statistic
        hit = ((test["预测超额"] > 0) == (test["建模标签"] > 0)).mean()
        top = test.nlargest(5, "预测超额")["建模标签"].mean()
        bottom = test.nsmallest(5, "预测超额")["建模标签"].mean()
        rows.append({
            "测试季度": quarter,
            "训练样本": len(train),
            "测试样本": len(test),
            "IC": ic,
            "方向命中率": hit,
            "Top5超额": top,
            "Bottom5超额": bottom,
            "Top5减Bottom5": top - bottom,
        })
        prediction_frames.append(test[["季度", "代码", "预测超额", "建模标签"]])

    if not rows:
        raise ValueError("No eligible out-of-sample quarters; check data and configuration")
    detail = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    residuals = predictions["建模标签"] - predictions["预测超额"]
    ic_std = detail["IC"].std(ddof=0)
    summary = {
        "oos_quarters": int(len(detail)),
        "ic_mean": float(detail["IC"].mean()),
        "ic_std_population": float(ic_std),
        "icir": float(detail["IC"].mean() / ic_std) if ic_std > 0 else np.nan,
        "hit_rate_quarter_average": float(detail["方向命中率"].mean()),
        "top5_bottom5_quarter_average": float(detail["Top5减Bottom5"].mean()),
        "oos_rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "oos_mae": float(residuals.abs().mean()),
        "oos_residual_std": float(residuals.std(ddof=1)),
    }
    return detail, predictions, summary
