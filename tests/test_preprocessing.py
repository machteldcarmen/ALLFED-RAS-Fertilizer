"""Tests for preprocessing helpers (shock application)."""

from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.model import FertilizerRAS  # noqa: E402
from src.preprocessing import (  # noqa: E402
    apply_trade_shock,
    apply_trade_shock_reported,
)


@pytest.fixture
def toy_trade():
    countries = ["Russia", "China", "USA", "India", "Brazil"]
    data = {
        "Russia": [0, 500, 300, 1200, 800],
        "China": [100, 0, 200, 1500, 400],
        "USA": [200, 300, 0, 400, 800],
        "India": [0, 50, 0, 0, 0],
        "Brazil": [0, 0, 50, 0, 0],
    }
    return pd.DataFrame(data, index=countries, columns=countries).T.astype(float)


def test_trade_shock_scales_touching_cells_only(toy_trade):
    T0 = toy_trade
    T_out = apply_trade_shock(T0, ["Russia"], 0.4)

    assert T_out.loc["Russia", "China"] == pytest.approx(500 * 0.4)
    assert T_out.loc["China", "Russia"] == pytest.approx(100 * 0.4)
    assert T_out.loc["China", "USA"] == pytest.approx(T0.loc["China", "USA"])
    assert T_out.loc["Russia", "Russia"] == 0.0
    assert T_out.loc["USA", "USA"] == 0.0


def test_trade_shock_no_double_scaling_on_bilateral(toy_trade):
    T0 = toy_trade
    T_out = apply_trade_shock(T0, ["Russia", "USA"], 0.4)

    assert T_out.loc["Russia", "USA"] == pytest.approx(T0.loc["Russia", "USA"] * 0.4)
    assert T_out.loc["USA", "Russia"] == pytest.approx(T0.loc["USA", "Russia"] * 0.4)


def test_trade_shock_ignores_unknown_countries(toy_trade):
    T0 = toy_trade
    T_out, report = apply_trade_shock_reported(T0, ["Russia", "Atlantis"], 0.4)

    assert report.matched_countries == ["Russia"]
    assert report.unmatched == ["Atlantis"]
    assert T_out.loc["Russia", "China"] == pytest.approx(500 * 0.4)


def test_trade_shock_reported_volumes(toy_trade):
    T0 = toy_trade
    mask = pd.DataFrame(False, index=T0.index, columns=T0.columns)
    mask.loc["Russia", :] = True
    mask.loc[:, "Russia"] = True
    volume_before = float((T0.values * mask.values).sum())

    _, report = apply_trade_shock_reported(T0, ["Russia"], 0.4)

    assert report.volume_before == pytest.approx(volume_before)
    assert report.volume_after == pytest.approx(volume_before * 0.4)


def test_trade_shock_changes_ras_flow_distribution(toy_trade):
    """Trade shock alters bilateral allocation even when row totals may match."""
    countries = toy_trade.index.tolist()
    P = pd.Series([4_000, 7_000, 5_500, 2_500, 1_000], index=countries, dtype=float)
    C = pd.Series([1_200, 6_500, 4_000, 5_000, 3_500], index=countries, dtype=float)
    T0 = toy_trade

    P_shocked = P.copy()
    P_shocked["Russia"] *= 0.4
    P_shocked["USA"] *= 0.4

    T0_reduced = apply_trade_shock(T0, ["Russia", "USA"], 0.4)

    baseline = FertilizerRAS(P_shocked, C, T0).run()
    trade_reduced = FertilizerRAS(P_shocked, C, T0_reduced).run()

    assert not baseline.X.equals(trade_reduced.X)
