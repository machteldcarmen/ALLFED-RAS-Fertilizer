"""Tests for the RAS model.

These tests cover the numerical invariants the model is supposed to
guarantee:

1. ``run_ras`` returns a matrix whose row sums equal S_hat and column
   sums equal D_hat (within tolerance).
2. Phase 1 satisfies per-country mass balance (K + S* = P or K + 0 = P
   for deficits).
3. ``FertilizerRAS.run`` returns F = K + X.sum(axis=0).
4. Degenerate inputs (zero production, zero demand, zero T0) are
   handled without crashing.

These tests run in well under a second — no FAOSTAT data needed.
"""

from __future__ import annotations

import sys
import pathlib

# Make ``src`` importable without installing the package.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.model import FertilizerRAS, run_ras


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def toy():
    """5-country toy example matching the README quickstart."""
    countries = ["Russia", "China", "USA", "India", "Brazil"]
    P = pd.Series([4_000, 7_000, 5_500, 2_500, 1_000], index=countries, dtype=float)
    C = pd.Series([1_200, 6_500, 4_000, 5_000, 3_500], index=countries, dtype=float)
    T0_data = {
        "Russia": [    0,  500,  300, 1200,  800],
        "China":  [  100,    0,  200, 1500,  400],
        "USA":    [  200,  300,    0,  400,  800],
        "India":  [    0,   50,    0,    0,    0],
        "Brazil": [    0,    0,   50,    0,    0],
    }
    T0 = pd.DataFrame(T0_data, index=countries, columns=countries).T.astype(float)
    return P, C, T0


# ──────────────────────────────────────────────────────────────────────────────
# run_ras
# ──────────────────────────────────────────────────────────────────────────────
def test_run_ras_matches_targets(toy):
    """Row and column sums of X must match S_hat and D_hat within tol."""
    _, _, T0 = toy
    countries = T0.index.tolist()
    S_hat = pd.Series([500.0, 200.0, 500.0, 0.0, 0.0], index=countries)
    D_hat = pd.Series([100.0, 150.0, 150.0, 400.0, 400.0], index=countries)
    # Make sums match exactly.
    S_hat = S_hat * (D_hat.sum() / S_hat.sum())

    X = run_ras(T0, S_hat, D_hat, tol=1e-8)

    assert np.allclose(X.sum(axis=1).values, S_hat.values, atol=1e-6)
    assert np.allclose(X.sum(axis=0).values, D_hat.values, atol=1e-6)


def test_run_ras_preserves_zero_structure(toy):
    """Zero entries in T0 must remain zero in X."""
    _, _, T0 = toy
    S_hat = pd.Series([300.0, 200.0, 500.0, 0.0, 0.0], index=T0.index)
    D_hat = pd.Series([ 50.0,  50.0, 100.0, 400.0, 400.0], index=T0.index)
    S_hat = S_hat * (D_hat.sum() / S_hat.sum())

    X = run_ras(T0, S_hat, D_hat)

    # Zeros in T0 must remain zeros in X. (RAS may additionally drive rows /
    # columns to zero when their target is 0, so the converse is NOT required.)
    assert np.all(X.values[T0.values == 0] == 0)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 mass balance
# ──────────────────────────────────────────────────────────────────────────────
def test_phase1_mass_balance(toy):
    """For every country: K + S* + (C - K - anything_else) is consistent."""
    P, C, T0 = toy
    model = FertilizerRAS(P, C, T0)
    K, S_star, D_star = model._phase1()

    B = P - C
    for c in P.index:
        if B[c] > 0:
            assert K[c] == pytest.approx(C[c])
            assert S_star[c] == pytest.approx(B[c])
            assert D_star[c] == pytest.approx(0.0)
        else:
            assert K[c] == pytest.approx(P[c])
            assert S_star[c] == pytest.approx(0.0)
            assert D_star[c] == pytest.approx(abs(B[c]))


# ──────────────────────────────────────────────────────────────────────────────
# Full run
# ──────────────────────────────────────────────────────────────────────────────
def test_full_run_F_equals_K_plus_imports(toy):
    P, C, T0 = toy
    res = FertilizerRAS(P, C, T0).run()

    recomputed_F = res.K + res.X.sum(axis=0)
    assert np.allclose(res.F.values, recomputed_F.values, atol=1e-9)


def test_full_run_hats_consistent(toy):
    P, C, T0 = toy
    res = FertilizerRAS(P, C, T0).run()

    assert res.S_hat.sum() == pytest.approx(res.D_hat.sum(), rel=1e-9)
    assert np.allclose(res.X.sum(axis=1).values, res.S_hat.values, atol=1e-4)
    assert np.allclose(res.X.sum(axis=0).values, res.D_hat.values, atol=1e-4)


def test_baseline_vs_shocked_reduces_trade(toy):
    P, C, T0 = toy
    model = FertilizerRAS(P, C, T0)
    baseline, shocked = model.run_shocked({"Russia": 0.4, "China": 0.7})

    # Total availability cannot exceed total production.
    assert shocked.F.sum() <= shocked.P.sum() + 1e-6
    # Shock reduced the global production total.
    assert shocked.P.sum() < baseline.P.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Degenerate inputs
# ──────────────────────────────────────────────────────────────────────────────
def test_zero_production_returns_only_domestic(toy):
    """If nobody produces, F must be 0 everywhere."""
    P, C, T0 = toy
    P0 = P * 0.0
    res = FertilizerRAS(P0, C, T0).run()
    assert res.X.values.sum() == pytest.approx(0.0)
    assert res.F.sum() == pytest.approx(0.0)


def test_zero_demand_returns_no_imports(toy):
    """If nobody consumes, there should be no imports and F == K == 0."""
    P, C, T0 = toy
    C0 = C * 0.0
    res = FertilizerRAS(P, C0, T0).run()
    # With zero demand, all production is "surplus" but nobody imports.
    # Therefore X must be zero and F == K == 0.
    assert res.X.values.sum() == pytest.approx(0.0)
    assert res.F.sum() == pytest.approx(0.0)


def test_zero_T0_returns_no_trade(toy):
    """With no historical trade, RAS has no structure to work with: X == 0."""
    P, C, T0 = toy
    T0_zero = T0 * 0.0
    res = FertilizerRAS(P, C, T0_zero).run()
    assert res.X.values.sum() == pytest.approx(0.0)
    # Each country only gets what it kept for itself.
    assert np.allclose(res.F.values, res.K.values)
