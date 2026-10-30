import numpy as np
import pandas as pd


def test_incomplete_future_labels_are_masked(prepared):
    q3 = prepared[prepared["季度"] == pd.Timestamp("2026-09-30")]
    q2 = prepared[prepared["季度"] == pd.Timestamp("2026-06-30")]
    assert len(q3) == 140
    assert q3["建模标签"].isna().all()
    assert q2["建模标签"].notna().all()
    assert (prepared["标签季度末"] <= pd.Timestamp("2026-10-30")).equals(
        prepared["标签是否完整"]
    )


def test_stale_fundamentals_are_invalidated(prepared):
    stale = prepared[prepared["财报是否陈旧"]]
    assert len(stale) == 116
    assert stale[["3C成长", "3C质量", "E/P"]].isna().all().all()
    dated = prepared.dropna(subset=["最新申报日"])
    assert (dated["最新申报日"] <= dated["季度"]).all()


def test_cross_sectional_zscores_are_centered(prepared, config):
    for feature in config.model.features:
        means = prepared.groupby("季度")[f"{feature}_z"].mean().dropna()
        assert np.allclose(means, 0.0, atol=1e-10)


def test_panel_keys_are_unique(prepared):
    assert not prepared.duplicated(["季度", "代码"]).any()
