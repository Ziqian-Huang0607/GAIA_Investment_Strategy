import pandas as pd
import pytest


def test_expanding_window_matches_expected_history(backtest_result):
    detail, predictions, summary = backtest_result
    assert len(detail) == 9
    assert detail.iloc[0]["测试季度"] == pd.Timestamp("2024-06-30")
    assert detail.iloc[0]["训练样本"] == 145
    assert detail.iloc[-1]["测试季度"] == pd.Timestamp("2026-06-30")
    assert detail.iloc[-1]["训练样本"] == 545
    assert len(predictions) == sum(detail["测试样本"])


def test_rebuilt_metrics_are_reproducible(backtest_result):
    _, _, summary = backtest_result
    assert summary["ic_mean"] == pytest.approx(0.06185489580447564, abs=1e-12)
    assert summary["hit_rate_quarter_average"] == pytest.approx(0.554527588813303, abs=1e-12)
    assert summary["top5_bottom5_quarter_average"] == pytest.approx(0.030118077832104406, abs=1e-12)
    assert summary["oos_rmse"] == pytest.approx(0.129796771433874, abs=1e-12)


def test_latest_model_uses_only_completed_training_quarters(latest_result):
    latest, coefficients, train_rows, score_quarter = latest_result
    assert train_rows == 595
    assert score_quarter == pd.Timestamp("2026-09-30")
    assert latest["预测下季度超额"].notna().sum() == 52
    assert len(coefficients) == 5
    assert (coefficients.iloc[:4]["coefficient"] >= 0).all()
