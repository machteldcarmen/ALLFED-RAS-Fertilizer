"""Data loading helpers.

Everything in this module is I/O: it takes paths to FAOSTAT bulk CSV
extracts and returns tidy ``pandas`` objects shaped for
:class:`src.model.FertilizerRAS`.

The preprocessing pipeline is intentionally thin so that it can be
replaced easily — if you have your production / demand / trade matrices
from another source, you can skip this module and feed ``FertilizerRAS``
directly.

Notes on FAOSTAT domains used
-----------------------------
* **Inputs — Fertilizers by Nutrient**
  (``Inputs_FertilizersNutrient_E_All_Data_NOFLAG.csv``) provides per-country
  production and agricultural-use quantities by nutrient.
* **Fertilizers — Detailed Trade Matrix**
  (``Fertilizers_DetailedTradeMatrix_E_All_Data_NOFLAG.csv``) provides
  bilateral export flows by nutrient.

Item codes used (FAOSTAT v1):
    N: 3102   P: 3103   K: 3104
Element codes used:
    Production 5510   Import 5610   Export 5910   AgUse 5157
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


NUTRIENT_CODES: dict[str, int] = {"N": 3102, "P": 3103, "K": 3104}
NUTRIENT_NAMES: dict[str, str] = {
    "N": "Nitrogen (N)",
    "P": "Phosphate (P2O5)",
    "K": "Potash (K2O)",
}
ELEMENT_CODES: dict[str, int] = {
    "Production": 5510,
    "Import": 5610,
    "Export": 5910,
    "AgUse": 5157,
}


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────
def _avg_over_years(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Add an ``avg`` column equal to the mean of ``Y{year}`` columns."""
    cols = [f"Y{y}" for y in years if f"Y{y}" in df.columns]
    if not cols:
        raise ValueError(f"No year columns {years} found in dataframe.")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["avg"] = df[cols].mean(axis=1)
    return df


def load_inputs(path: str | Path, years: list[int]) -> pd.DataFrame:
    """Load the FAOSTAT *Inputs — Fertilizers by Nutrient* CSV.

    Returns the raw dataframe with an added ``avg`` column averaged over
    ``years``.  This is then consumed by :func:`get_production_demand`.
    """
    df = pd.read_csv(path, low_memory=False)
    return _avg_over_years(df, years)


def load_trade(path: str | Path, years: list[int]) -> pd.DataFrame:
    """Load the FAOSTAT *Detailed Trade Matrix* CSV.

    Returns the raw dataframe with an added ``avg`` column averaged over
    ``years``.  This is then consumed by :func:`get_trade_matrix`.
    """
    df = pd.read_csv(path, low_memory=False)
    return _avg_over_years(df, years)


# ──────────────────────────────────────────────────────────────────────────────
# Extractors (per nutrient)
# ──────────────────────────────────────────────────────────────────────────────
def _series_by_code(df: pd.DataFrame, item_code: int, element_code: int) -> pd.Series:
    mask = (df["Item Code"] == item_code) & (df["Element Code"] == element_code)
    s = df.loc[mask, ["Area", "avg"]].dropna(subset=["avg"])
    return s.groupby("Area")["avg"].sum()


def get_production_demand(
    inputs_df: pd.DataFrame, nutrient: str
) -> tuple[pd.Series, pd.Series]:
    """Extract (production, agricultural-use demand) series for one nutrient.

    Both series are indexed by FAOSTAT *Area* names and expressed in tonnes
    (FAOSTAT unit for *Nutrients*).
    """
    if nutrient not in NUTRIENT_CODES:
        raise ValueError(
            f"Unknown nutrient {nutrient!r}; expected one of {list(NUTRIENT_CODES)}"
        )
    ic = NUTRIENT_CODES[nutrient]
    P = _series_by_code(inputs_df, ic, ELEMENT_CODES["Production"])
    C = _series_by_code(inputs_df, ic, ELEMENT_CODES["AgUse"])
    P.name = f"Production_{nutrient}"
    C.name = f"Demand_{nutrient}"
    return P, C


def get_trade_matrix(trade_df: pd.DataFrame, nutrient: str) -> pd.DataFrame:
    """Extract the bilateral export trade matrix for one nutrient.

    Rows are *Reporter Countries* (exporters), columns are
    *Partner Countries* (importers).  Only Export flows are kept; zero
    or negative values are dropped.
    """
    if nutrient not in NUTRIENT_CODES:
        raise ValueError(
            f"Unknown nutrient {nutrient!r}; expected one of {list(NUTRIENT_CODES)}"
        )
    ic = NUTRIENT_CODES[nutrient]
    mask = (trade_df["Item Code"] == ic) & (
        trade_df["Element Code"] == ELEMENT_CODES["Export"]
    )
    flows = trade_df.loc[
        mask, ["Reporter Countries", "Partner Countries", "avg"]
    ].dropna(subset=["avg"])
    flows = flows[flows["avg"] > 0]
    return flows.pivot_table(
        index="Reporter Countries",
        columns="Partner Countries",
        values="avg",
        aggfunc="sum",
        fill_value=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Country alignment / filtering
# ──────────────────────────────────────────────────────────────────────────────
def filter_countries(
    P: pd.Series,
    C: pd.Series,
    T0: pd.DataFrame,
    min_threshold: float = 1_000.0,
) -> tuple[list[str], pd.Series, pd.Series, pd.DataFrame]:
    """Intersect/align indices and drop negligible countries.

    A country is kept iff its production **or** demand is at least
    ``min_threshold``.  The trade matrix is reindexed to that set and its
    diagonal is zeroed.  All three outputs share the same sorted index.

    Returns
    -------
    countries, P, C, T0
    """
    all_c = sorted(set(P.index) | set(C.index))
    P_full = P.reindex(all_c, fill_value=0).clip(lower=0)
    C_full = C.reindex(all_c, fill_value=0).clip(lower=0)

    keep = (P_full >= min_threshold) | (C_full >= min_threshold)
    countries = sorted(P_full[keep].index)

    P_out = P_full.reindex(countries, fill_value=0)
    C_out = C_full.reindex(countries, fill_value=0)

    T0_out = T0.reindex(index=countries, columns=countries, fill_value=0).astype(float)
    vals = T0_out.to_numpy(copy=True)
    np.fill_diagonal(vals, 0.0)
    T0_out = pd.DataFrame(vals, index=countries, columns=countries)

    return countries, P_out, C_out, T0_out


# ──────────────────────────────────────────────────────────────────────────────
# Shock application
# ──────────────────────────────────────────────────────────────────────────────
def apply_shock(P: pd.Series, shock: dict[str, float]) -> pd.Series:
    """Return a new production vector with ``P[c] *= factor`` for ``c`` in ``shock``.

    ``shock`` maps country name to *surviving fraction*
    (``0.4`` = 60 % cut).  Countries not in ``shock`` are unchanged.
    Missing country names are ignored silently but recorded in
    :attr:`ShockReport.unmatched` if you use :func:`apply_shock_reported`.
    """
    P_out = P.copy()
    for country, factor in shock.items():
        if country in P_out.index:
            P_out[country] = P[country] * factor
    return P_out


@dataclass
class ShockReport:
    """Structured record of what ``apply_shock_reported`` actually changed."""

    applied: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    def as_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "country": c,
                "P_before": before,
                "P_after": after,
                "surviving_fraction": frac,
            }
            for c, (before, after, frac) in self.applied.items()
        ]
        return pd.DataFrame(rows)


def apply_shock_reported(
    P: pd.Series, shock: dict[str, float]
) -> tuple[pd.Series, ShockReport]:
    """Same as :func:`apply_shock` but also returns what was (un)applied."""
    P_out = P.copy()
    report = ShockReport()
    for country, factor in shock.items():
        if country in P_out.index:
            before = float(P[country])
            after = float(before * factor)
            P_out[country] = after
            report.applied[country] = (before, after, factor)
        else:
            report.unmatched.append(country)
    return P_out, report


# ──────────────────────────────────────────────────────────────────────────────
# One-shot convenience: load everything for one nutrient
# ──────────────────────────────────────────────────────────────────────────────
def load_nutrient(
    inputs_path: str | Path,
    trade_path: str | Path,
    nutrient: str,
    years: list[int],
    min_threshold: float = 1_000.0,
) -> dict[str, object]:
    """Load FAOSTAT data and return ``{countries, P, C, T0}`` for one nutrient.

    The output of this function is exactly what
    :class:`src.model.FertilizerRAS` expects.
    """
    inputs_df = load_inputs(inputs_path, years)
    trade_df = load_trade(trade_path, years)

    P, C = get_production_demand(inputs_df, nutrient)
    T0 = get_trade_matrix(trade_df, nutrient)

    countries, P, C, T0 = filter_countries(P, C, T0, min_threshold=min_threshold)
    return {"countries": countries, "P": P, "C": C, "T0": T0}
