"""Post-processing: compare runs, summarize, and write results to disk.

Everything here operates on :class:`src.model.RASResult` objects produced
by :class:`src.model.FertilizerRAS.run` and returns plain ``pandas``
objects (no plotting — see :mod:`src.utils` for that).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import RASResult


# ──────────────────────────────────────────────────────────────────────────────
# Baseline vs shocked comparison
# ──────────────────────────────────────────────────────────────────────────────
def build_comparison(
    baseline: RASResult,
    shocked: RASResult,
    drop_zero_demand: bool = True,
) -> pd.DataFrame:
    """Build a side-by-side comparison table of baseline vs shocked.

    The two results must be for the same country index (this is the
    case when they come from :meth:`FertilizerRAS.run_shocked`).
    """
    df = pd.DataFrame(
        {
            "P_baseline": baseline.P,
            "P_shocked": shocked.P,
            "Demand": baseline.C,
            "F_baseline": baseline.F,
            "F_shocked": shocked.F,
            "Cover_base_%": baseline.coverage,
            "Cover_shock_%": shocked.coverage,
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    if drop_zero_demand:
        df = df[df["Demand"].fillna(0) > 0]

    df["Change_pp"] = df["Cover_shock_%"] - df["Cover_base_%"]
    df["Avail_vs_baseline_%"] = (
        df["F_shocked"] / df["F_baseline"].replace(0, np.nan) * 100
    ).round(1)
    df["Avail_reduction_%"] = (100 - df["Avail_vs_baseline_%"]).round(1)
    return df.round(2)


def most_affected(comparison: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return the ``n`` countries with the worst coverage drop (most negative)."""
    return comparison.nsmallest(n, "Change_pp")


def least_affected(comparison: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the ``n`` countries whose coverage changed the least or improved."""
    return comparison.nlargest(n, "Change_pp")


# ──────────────────────────────────────────────────────────────────────────────
# Global / sanity summaries
# ──────────────────────────────────────────────────────────────────────────────
def global_summary(baseline: RASResult, shocked: RASResult) -> pd.DataFrame:
    """Global totals: production, demand, trade, availability — baseline vs shocked."""
    rows = {
        "Total production": (baseline.P.sum(), shocked.P.sum()),
        "Total demand": (baseline.C.sum(), shocked.C.sum()),
        "Total trade (sum X)": (
            float(baseline.X.values.sum()),
            float(shocked.X.values.sum()),
        ),
        "Total availability (sum F)": (baseline.F.sum(), shocked.F.sum()),
        "Global coverage %": (
            baseline.F.sum() / max(baseline.C.sum(), 1e-12) * 100,
            shocked.F.sum() / max(shocked.C.sum(), 1e-12) * 100,
        ),
    }
    df = pd.DataFrame(rows, index=["baseline", "shocked"]).T
    df["change"] = df["shocked"] - df["baseline"]
    return df.round(2)


def sanity_checks(result: RASResult, tol: float = 1.0) -> pd.DataFrame:
    """Verify RAS mass balances.

    Returns a one-row DataFrame with max row-sum error, max col-sum error,
    total-X, sum(F), sum(K)+sum(X), and a ``ok`` column indicating whether
    all checks are within ``tol`` (same units as the input data).
    """
    row_err = float((result.X.sum(axis=1) - result.S_hat).abs().max())
    col_err = float((result.X.sum(axis=0) - result.D_hat).abs().max())
    sum_X = float(result.X.values.sum())
    sum_F = float(result.F.sum())
    sum_K_plus_X = float(result.K.sum()) + sum_X

    return pd.DataFrame(
        [
            {
                "max_row_error": row_err,
                "max_col_error": col_err,
                "sum_X": sum_X,
                "sum_F": sum_F,
                "sum_K_plus_X": sum_K_plus_X,
                "F_eq_K_plus_X": abs(sum_F - sum_K_plus_X) < tol,
                "rows_ok": row_err < tol,
                "cols_ok": col_err < tol,
            }
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Export to disk
# ──────────────────────────────────────────────────────────────────────────────
def save_result(
    result: RASResult,
    out_dir: str | Path,
    tag: str,
) -> dict[str, Path]:
    """Write ``X`` (matrix) and per-country summary to ``out_dir``.

    Two CSV files are created:

    * ``{out_dir}/X_{tag}.csv``      — bilateral trade matrix
    * ``{out_dir}/summary_{tag}.csv`` — per-country summary

    ``tag`` typically looks like ``"N_baseline"`` or ``"K_shocked"``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = out_dir / f"X_{tag}.csv"
    s_path = out_dir / f"summary_{tag}.csv"
    result.X.to_csv(x_path)
    result.summary().to_csv(s_path)
    return {"X": x_path, "summary": s_path}
