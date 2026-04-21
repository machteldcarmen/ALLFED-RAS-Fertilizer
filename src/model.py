"""4-phase fertilizer-trade RAS (iterative proportional fitting) model.

The model takes a post-shock *production* vector ``P``, a *demand* vector
``C``, and a *historical* bilateral trade matrix ``T0`` and returns

* a post-shock bilateral trade matrix ``X`` (row sums = feasible export
  targets, column sums = feasible import targets, with the same
  *structure* as ``T0``), and
* a final per-country fertilizer availability vector
  ``F_i = K_i + sum_j x_{ji}``.

The full mathematical description (equations 1-14, notation, references)
lives in ``docs/methodology.md`` of this repository.  This module is the
single source of truth for the numerical implementation.

The model runs **independently** for each nutrient (N, P, K) — there is
no cross-nutrient coupling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — RAS / Iterative Proportional Fitting
# ──────────────────────────────────────────────────────────────────────────────
def run_ras(
    T0: pd.DataFrame,
    S_hat: pd.Series,
    D_hat: pd.Series,
    max_iter: int = 1_000,
    tol: float = 1e-6,
    verbose: bool = False,
    epsilon: float = 1e-9,
    x_history: list[pd.DataFrame] | None = None,
    r_history: list[pd.Series] | None = None,
    c_history: list[pd.Series] | None = None,
) -> pd.DataFrame:
    """Find a matrix ``X`` that matches given row/col totals and preserves structure.

    Solves the standard bi-proportional fitting problem

        x_ij = r_i * T0_ij * c_j

    such that

        sum_j x_ij = S_hat_i    (row / export totals)
        sum_i x_ij = D_hat_j    (column / import totals)

    by alternately rescaling rows and columns until ``max(row_err, col_err)``
    is below ``tol``.

    **Algebraic simplification.**  Because
    ``row_sums_i = r_i * sum_j T_ij * c_j``, the update
    ``r_new_i = r_i * s_i / row_sums_i`` simplifies to
    ``r_new_i = s_i / (T c)_i`` — the old ``r_i`` cancels exactly.  The same
    holds for the column update.  We therefore only compute the matrix-vector
    products ``T @ c`` and ``r @ T`` per iteration and build the full
    ``N x N`` matrix ``X`` only once, at the end.  This keeps the hot loop at
    ``O(N^2)`` memory traffic instead of three ``N x N`` allocations per
    iteration and runs on NumPy's BLAS backend.

    **Structural zero / "epsilon seed".**  ALLFED shock scenarios can create
    situations where a country suddenly needs to export (or import) even
    though every historical entry in its row (or column) of ``T0`` is
    exactly zero.  Pure RAS is multiplicative, so a zero row in ``T0`` stays
    zero forever and the algorithm cannot place those flows.  To allow such
    *emergency* trade routes to open, every off-diagonal zero entry in
    ``T0`` is replaced by a tiny ``epsilon`` (default ``1e-9``).  The
    diagonal is preserved at zero (countries cannot self-trade).  Set
    ``epsilon=0.0`` to disable this behavior and keep the structural zeros.

    Parameters
    ----------
    T0
        Historical trade matrix.  ``T0.loc[i, j]`` is the export from country
        ``i`` to country ``j``.  Index and columns must match and be in the
        same order.
    S_hat, D_hat
        Feasible export / import targets (see :func:`FertilizerRAS._phase2`).
        Their sums must match up to numerical precision; otherwise the
        algorithm cannot converge.
    max_iter, tol, verbose
        Iteration cap, convergence tolerance on the maximum row/column-sum
        error (same units as ``T0``), and whether to print the iteration
        on which we converged.
    epsilon
        Seed value used to replace off-diagonal *structural zeros* in
        ``T0``.  Defaults to ``1e-9``.  Use ``0.0`` to keep the strict
        historical zero pattern.
    x_history
        If given, append a **copy** of the current fitted matrix
        ``X = diag(r) @ T @ diag(c)`` after **each** RAS iteration (after
        both the row and column multipliers are updated).  Useful for small
        teaching examples; leave ``None`` in large runs to avoid ``O(k N^2)``
        memory.  Degenerate runs (no trade) append nothing.
    r_history, c_history
        If given, append a copy of the row-multiplier vector ``r`` (right
        after the row-update step) and of the column-multiplier vector
        ``c`` (right after the column-update step) for **every** RAS
        iteration.  Each entry is a ``pd.Series`` indexed by country.
        Same memory caveat as ``x_history``: only use for small matrices.

    Returns
    -------
    pd.DataFrame
        The fitted trade matrix ``X`` with the same index/columns as ``T0``.

    Warns
    -----
    RuntimeWarning
        If the iteration cap is reached without converging below ``tol``.
        This is emitted unconditionally (even when ``verbose=False``) so
        Monte-Carlo pipelines do not fail silently.
    """
    if not (T0.index.equals(T0.columns)):
        raise ValueError("T0 must be a square matrix with identical index and columns.")
    if not T0.index.equals(S_hat.index) or not T0.index.equals(D_hat.index):
        raise ValueError("T0, S_hat and D_hat must share the same country index.")

    countries = T0.index.tolist()
    T = T0.fillna(0.0).to_numpy(dtype=float, copy=True)

    if epsilon > 0:
        zero_mask = T == 0
        np.fill_diagonal(zero_mask, False)
        T[zero_mask] = epsilon

    s = S_hat.to_numpy(dtype=float, copy=True)
    d = D_hat.to_numpy(dtype=float, copy=True)

    n = len(countries)
    r = np.ones(n)
    c = np.ones(n)

    Tc = T @ c
    max_err = np.inf

    for iteration in range(1, max_iter + 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(Tc > 0, s / Tc, r)

        if r_history is not None:
            r_history.append(pd.Series(r.copy(), index=countries, name=f"r_iter{iteration}"))

        rT = r @ T
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(rT > 0, d / rT, c)

        if c_history is not None:
            c_history.append(pd.Series(c.copy(), index=countries, name=f"c_iter{iteration}"))

        Tc = T @ c
        row_err = np.abs(r * Tc - s).max()
        col_err = np.abs(c * rT - d).max()
        max_err = float(max(row_err, col_err))

        if x_history is not None:
            X_iter = r[:, None] * T * c[None, :]
            x_history.append(
                pd.DataFrame(X_iter, index=countries, columns=countries).copy()
            )

        if max_err < tol:
            if verbose:
                print(
                    f"RAS converged in {iteration} iterations (max_err={max_err:.2e})"
                )
            break
    else:
        msg = (
            f"RAS did NOT converge in {max_iter} iterations "
            f"(max_err={max_err:.2e})"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if verbose:
            print(msg)

    X_final = r[:, None] * T * c[None, :]
    return pd.DataFrame(X_final, index=countries, columns=countries)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RASResult:
    """Per-country outputs of one RAS run.

    Attributes
    ----------
    P, C
        Production and demand vectors used for this run (possibly shocked).
    T0
        Historical trade matrix used as the structural prior.
    K
        Domestic production kept for local use (Phase 1).
    S_star, D_star
        Target export / import (before global consistency scaling, Phase 1).
    S_hat, D_hat
        Feasible export / import targets after Phase 2 (their sums agree).
    X
        Post-shock bilateral trade matrix (Phase 3 output).
    F
        Final fertilizer availability per country (Phase 4 output),
        ``F_i = K_i + sum_j x_{ji}``.
    """

    P: pd.Series
    C: pd.Series
    T0: pd.DataFrame
    K: pd.Series
    S_star: pd.Series
    D_star: pd.Series
    S_hat: pd.Series
    D_hat: pd.Series
    X: pd.DataFrame
    F: pd.Series

    @property
    def imports_received(self) -> pd.Series:
        """Column sums of ``X`` — imports received per country."""
        s = self.X.sum(axis=0)
        s.name = "imports_received"
        return s

    @property
    def exports_sent(self) -> pd.Series:
        """Row sums of ``X`` — exports sent per country."""
        s = self.X.sum(axis=1)
        s.name = "exports_sent"
        return s

    @property
    def coverage(self) -> pd.Series:
        """``F / C`` in percent, where ``C > 0``; NaN elsewhere."""
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = (self.F / self.C.replace(0, np.nan)) * 100
        cov.name = "coverage_%"
        return cov

    def summary(self) -> pd.DataFrame:
        """Per-country overview table (P, C, K, imports, F, coverage, unmet)."""
        unmet = (self.C - self.F).clip(lower=0)
        unmet.name = "unmet_demand"
        return pd.DataFrame(
            {
                "Production_P": self.P,
                "Demand_C": self.C,
                "Kept_K": self.K,
                "Imports_received": self.imports_received,
                "F_final": self.F,
                "Unmet_demand": unmet,
                "Coverage_%": self.coverage.round(1),
            }
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model interface
# ──────────────────────────────────────────────────────────────────────────────
class FertilizerRAS:
    """Fertilizer trade inertia model (4-phase RAS pipeline).

    A single instance ties one set of inputs (``P``, ``C``, ``T0``) to one
    RAS run.  Running the model for multiple nutrients is done by creating
    one instance per nutrient; nutrients do not interact.

    Parameters
    ----------
    P
        Production vector (country -> kt or t, units only need to be
        consistent across ``P``, ``C`` and ``T0``).  Represents the
        *post-shock* production (apply any shocks before constructing
        the model — see :mod:`src.preprocessing`).
    C
        Domestic demand vector (country -> same units as ``P``).
    T0
        Historical bilateral trade matrix with matching index/columns.
        ``T0.loc[i, j]`` = exports from country ``i`` to country ``j``.
        The diagonal is ignored (countries cannot self-trade); the
        constructor zeroes it out defensively.
    max_iter, tol
        RAS iteration cap and convergence tolerance (see :func:`run_ras`).
    epsilon
        Seed used by :func:`run_ras` to replace off-diagonal *structural
        zeros* in ``T0``.  Allows emergency trade routes to open in
        post-shock scenarios where a historically-isolated country must
        suddenly export or import.  Set to ``0.0`` to disable.

    Example
    -------
    >>> model = FertilizerRAS(P, C, T0)
    >>> result = model.run()
    >>> print(result.summary().head())
    """

    def __init__(
        self,
        P: pd.Series,
        C: pd.Series,
        T0: pd.DataFrame,
        max_iter: int = 1_000,
        tol: float = 1e-6,
        epsilon: float = 1e-9,
    ) -> None:
        countries = self._align_inputs(P, C, T0)
        self.countries = countries
        self.P = P.reindex(countries, fill_value=0).astype(float)
        self.C = C.reindex(countries, fill_value=0).astype(float)

        T0 = T0.reindex(index=countries, columns=countries, fill_value=0).astype(float)
        vals = T0.to_numpy(copy=True)
        np.fill_diagonal(vals, 0.0)
        self.T0 = pd.DataFrame(vals, index=countries, columns=countries)

        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon

    @staticmethod
    def _align_inputs(P, C, T0) -> list[str]:
        """Shared sorted country list across all three inputs."""
        idx = set(P.index) | set(C.index) | set(T0.index) | set(T0.columns)
        return sorted(idx)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    def _phase1(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Domestic first filter.

        For each country ``i`` with net balance ``B_i = P_i - C_i``:

        * Surplus (``B_i > 0``):  ``K_i = C_i``, ``S*_i = B_i``, ``D*_i = 0``
        * Deficit (``B_i <= 0``): ``K_i = P_i``, ``S*_i = 0``, ``D*_i = |B_i|``
        """
        B = self.P - self.C
        surplus = B > 0

        K = self.P.where(~surplus, self.C)  # deficit: P, surplus: C
        S_star = B.where(surplus, 0.0).astype(float)
        D_star = (-B).where(~surplus, 0.0).astype(float)

        K.name, S_star.name, D_star.name = "K", "S_star", "D_star"
        return K, S_star, D_star

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    @staticmethod
    def _phase2(S_star: pd.Series, D_star: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Scale the larger side down so total supply == total demand.

        RAS requires ``sum(S_hat) == sum(D_hat)``.  If there is a global
        shortage, demand is scaled down; if there is a global surplus,
        supply is scaled down.
        """
        S_total, D_total = S_star.sum(), D_star.sum()
        S_hat, D_hat = S_star.copy(), D_star.copy()

        if S_total <= 0 or D_total <= 0:
            # Degenerate — no trade possible; keep zeros.
            S_hat[:] = 0.0
            D_hat[:] = 0.0
        elif S_total < D_total:
            D_hat = D_star * (S_total / D_total)
        elif S_total > D_total:
            S_hat = S_star * (D_total / S_total)
        # else equal already — nothing to do.

        S_hat.name, D_hat.name = "S_hat", "D_hat"
        return S_hat, D_hat

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(
        self,
        verbose: bool = False,
        x_history: list[pd.DataFrame] | None = None,
        r_history: list[pd.Series] | None = None,
        c_history: list[pd.Series] | None = None,
    ) -> RASResult:
        """Run Phases 1-4 and return all intermediate + final quantities.

        Parameters
        ----------
        verbose
            Print RAS convergence message when applicable.
        x_history, r_history, c_history
            Optional lists passed through to :func:`run_ras` — each RAS
            iteration appends a copy of the current ``X`` / row multipliers
            ``r`` / column multipliers ``c`` respectively.  Use for small
            diagnostics only.
        """
        K, S_star, D_star = self._phase1()
        S_hat, D_hat = self._phase2(S_star, D_star)

        if S_hat.sum() == 0 or D_hat.sum() == 0:
            # No trade flows possible; X is zeros.
            X = pd.DataFrame(0.0, index=self.countries, columns=self.countries)
        else:
            X = run_ras(
                self.T0,
                S_hat,
                D_hat,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=verbose,
                epsilon=self.epsilon,
                x_history=x_history,
                r_history=r_history,
                c_history=c_history,
            )

        F = K + X.sum(axis=0)
        F.name = "F_final"

        return RASResult(
            P=self.P,
            C=self.C,
            T0=self.T0,
            K=K,
            S_star=S_star,
            D_star=D_star,
            S_hat=S_hat,
            D_hat=D_hat,
            X=X,
            F=F,
        )

    # ── Convenience ───────────────────────────────────────────────────────────
    def run_shocked(
        self,
        shock: dict[str, float],
        verbose: bool = False,
    ) -> tuple[RASResult, RASResult]:
        """Run the model twice: once with ``self.P`` (baseline) and once with
        a shocked production vector.

        ``shock`` maps country name -> *surviving production fraction*
        (e.g. ``0.4`` = 60 % cut).  Countries not in ``shock`` keep their
        baseline production.

        Returns
        -------
        (baseline_result, shocked_result)
        """
        baseline = self.run(verbose=verbose)

        P_shocked = self.P.copy()
        for country, factor in shock.items():
            if country in P_shocked.index:
                P_shocked[country] = self.P[country] * factor

        shocked_model = FertilizerRAS(
            P_shocked,
            self.C,
            self.T0,
            max_iter=self.max_iter,
            tol=self.tol,
            epsilon=self.epsilon,
        )
        shocked = shocked_model.run(verbose=verbose)
        return baseline, shocked
