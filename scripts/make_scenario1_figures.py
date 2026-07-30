"""Generate figures for the US/Russia nuclear-war scenario (scenario 1).

Reads the per-country summary CSVs in ``results/`` and writes publication
figures to ``results/figures/``:

* ``fig1_coverage_map_{N,P,K}.png`` - choropleth of post-shock supply
  coverage (% of demand met) per country.
* ``fig1_coverage_map_liebig.png`` - choropleth of the *limiting* nutrient
  per country (min coverage across N, P, K; Liebig's law of the minimum).
* ``fig2_major_countries.png``      - grouped bars of post-shock coverage for
  the largest fertilizer consumers, by nutrient.
* ``fig3_global_summary.png``       - global coverage (baseline vs shock) and
  unmet demand per nutrient.

Run from the project root::

    python scripts/make_scenario1_figures.py
    python scripts/make_scenario1_figures.py --tag scenario1_us_ru_reduced_trade
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import pycountry

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
AREACODES = (
    ROOT / "data" / "Inputs_FertilizersNutrient_E_All_Data"
    / "Inputs_FertilizersNutrient_E_AreaCodes.csv"
)

NUTS = {"N": "Nitrogen (N)", "P": "Phosphate (P\u2082O\u2085)", "K": "Potash (K\u2082O)"}
DEFAULT_TAG = "scenario1_us_ru_nuclear"
TAG_SUBTITLES = {
    "scenario1_us_ru_nuclear": (
        "US/Russia nuclear-war scenario (US/RU \u221260%, rest of world \u221218% production)"
    ),
    "scenario1_us_ru_reduced_trade": (
        "US/Russia nuclear-war scenario 1b "
        "(US/RU \u221260% production, \u221260% trade on US/RU flows, rest \u221218% production)"
    ),
}


def _fig_suffix(tag: str) -> str:
    """Filename suffix: empty for the default scenario (backward compatible)."""
    return "" if tag == DEFAULT_TAG else f"_{tag}"


MAJORS = [
    ("China, mainland", "China"),
    ("India", "India"),
    ("United States of America", "United States"),
    ("Brazil", "Brazil"),
    ("Indonesia", "Indonesia"),
    ("Pakistan", "Pakistan"),
    ("Bangladesh", "Bangladesh"),
    ("Viet Nam", "Viet Nam"),
    ("France", "France"),
    ("Germany", "Germany"),
    ("Canada", "Canada"),
    ("Russian Federation", "Russia"),
    ("Nigeria", "Nigeria"),
    ("Ethiopia", "Ethiopia"),
]

# Colors
RED, AMBER, GREEN, BLUE, GREY = "#c0392b", "#e67e22", "#27ae60", "#2980b9", "#7f8c8d"


# ──────────────────────────────────────────────────────────────────────────────
def build_iso3_map() -> dict[str, str]:
    """FAOSTAT Area name -> ISO alpha-3 via the M49 numeric code."""
    ac = pd.read_csv(AREACODES)
    ac.columns = [c.strip() for c in ac.columns]
    name_col = "Area"
    m49_col = [c for c in ac.columns if "M49" in c][0]
    out: dict[str, str] = {}
    for _, row in ac.iterrows():
        name = str(row[name_col]).strip()
        m49 = str(row[m49_col]).strip().lstrip("'").lstrip("0") or "0"
        try:
            rec = pycountry.countries.get(numeric=f"{int(m49):03d}")
        except (ValueError, TypeError):
            rec = None
        if rec is not None:
            out[name] = rec.alpha_3
    # explicit fixes for FAOSTAT-specific labels
    out.setdefault("China, mainland", "CHN")
    out.setdefault("China, Taiwan Province of", "TWN")
    return out


def load_summary(code: str, kind: str, tag: str) -> pd.DataFrame:
    return pd.read_csv(RES / f"summary_{code}_{tag}_{kind}.csv", index_col=0)


# ──────────────────────────────────────────────────────────────────────────────
def fig_coverage_maps(iso3: dict[str, str], tag: str, subtitle: str) -> None:
    import plotly.graph_objects as go

    for code, label in NUTS.items():
        s = load_summary(code, "shocked", tag)
        s = s[s["Demand_C"].fillna(0) > 0]
        locs, z, txt = [], [], []
        for area, row in s.iterrows():
            c = iso3.get(str(area))
            if not c:
                continue
            locs.append(c)
            z.append(float(row["Coverage_%"]))
            txt.append(f"<b>{area}</b><br>Coverage: {row['Coverage_%']:.1f}%")
        fig = go.Figure(
            go.Choropleth(
                locations=locs, z=z, locationmode="ISO-3",
                colorscale="RdYlGn", zmin=40, zmax=100,
                marker_line_color="white", marker_line_width=0.4,
                colorbar_title="Coverage %", hovertext=txt, hoverinfo="text",
            )
        )
        fig.update_layout(
            title=(
                f"Post-shock fertilizer coverage \u2014 {label}"
                f"<br><sub>{subtitle}</sub>"
            ),
            geo=dict(showframe=False, showcoastlines=True,
                     projection_type="natural earth"),
            height=560, margin=dict(l=10, r=10, t=80, b=10),
        )
        out = FIG / f"fig1_coverage_map_{code}{_fig_suffix(tag)}.png"
        fig.write_image(str(out), scale=2)
        print(f"wrote {out}")


def fig_coverage_map_liebig(iso3: dict[str, str], tag: str, subtitle: str) -> None:
    """Choropleth of min(N, P, K) coverage per country (Liebig's law).

    Agriculture needs all three nutrients; the scarcest one limits effective
    fertilizer security, so we map the minimum post-shock coverage.
    """
    import plotly.graph_objects as go

    frames: dict[str, pd.Series] = {}
    for code in NUTS:
        s = load_summary(code, "shocked", tag)
        s = s[s["Demand_C"].fillna(0) > 0]
        frames[code] = s["Coverage_%"]

    combined = pd.DataFrame(frames)
    limiting = combined.min(axis=1, skipna=True)
    which = combined.idxmin(axis=1, skipna=True)

    locs, z, txt = [], [], []
    for area, val in limiting.items():
        c = iso3.get(str(area))
        if not c or pd.isna(val):
            continue
        locs.append(c)
        z.append(float(val))
        nut = which.get(area, "?")
        n = combined.loc[area, "N"] if "N" in combined.columns else np.nan
        p = combined.loc[area, "P"] if "P" in combined.columns else np.nan
        k = combined.loc[area, "K"] if "K" in combined.columns else np.nan
        txt.append(
            f"<b>{area}</b><br>"
            f"Limiting coverage: {val:.1f}% ({NUTS.get(nut, nut)})<br>"
            f"N: {n:.1f}% &nbsp; P: {p:.1f}% &nbsp; K: {k:.1f}%"
        )

    fig = go.Figure(
        go.Choropleth(
            locations=locs,
            z=z,
            locationmode="ISO-3",
            colorscale="RdYlGn",
            zmin=40,
            zmax=100,
            marker_line_color="white",
            marker_line_width=0.4,
            colorbar_title="Coverage %",
            hovertext=txt,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=(
            "Post-shock fertilizer coverage \u2014 limiting nutrient "
            "(Liebig\u2019s law of the minimum)"
            f"<br><sub>{subtitle}</sub>"
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
        ),
        height=560,
        margin=dict(l=10, r=10, t=80, b=10),
    )
    out = FIG / f"fig1_coverage_map_liebig{_fig_suffix(tag)}.png"
    fig.write_image(str(out), scale=2)
    print(f"wrote {out}")


def fig_major_countries(tag: str, subtitle: str) -> None:
    import matplotlib.pyplot as plt

    data = {code: load_summary(code, "shocked", tag) for code in NUTS}
    rows = []
    for fao, disp in MAJORS:
        rec = {"Country": disp}
        for code in NUTS:
            s = data[code]
            rec[code] = float(s.loc[fao, "Coverage_%"]) if fao in s.index else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("Country")
    df = df.loc[df["N"].sort_values().index]  # worst N at top

    y = np.arange(len(df))
    h = 0.26
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y + h, df["N"], height=h, color=BLUE, label="Nitrogen")
    ax.barh(y, df["P"], height=h, color=AMBER, label="Phosphate")
    ax.barh(y - h, df["K"], height=h, color=GREEN, label="Potash")
    ax.axvline(100, color=GREY, ls="--", lw=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Post-shock supply coverage (% of demand met)")
    ax.set_xlim(0, 110)
    ax.set_title(
        "Post-shock fertilizer coverage for major consumers\n"
        f"{subtitle}", fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG / f"fig2_major_countries{_fig_suffix(tag)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig_global_summary(tag: str, subtitle: str) -> None:
    import matplotlib.pyplot as plt

    labels, cov_shock, unmet_pct, unmet_mt = [], [], [], []
    for code, label in NUTS.items():
        b = load_summary(code, "baseline", tag)
        s = load_summary(code, "shocked", tag)
        dem = b["Demand_C"].sum()
        unmet = s["Unmet_demand"].sum()
        labels.append(label.split(" (")[0])
        cov_shock.append(s["F_final"].sum() / dem * 100)
        unmet_pct.append(unmet / dem * 100)
        unmet_mt.append(unmet / 1e6)

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, [100] * len(labels), color=GREY, alpha=0.35, label="Baseline (100%)")
    bars = ax1.bar(x, cov_shock, color=[RED, AMBER, GREEN], label="Post-shock")
    for xi, c in zip(x, cov_shock):
        ax1.text(xi, c + 1.5, f"{c:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Global supply coverage (%)")
    ax1.set_ylim(0, 110)
    ax1.set_title("Global coverage: baseline vs post-shock")
    ax1.legend(loc="lower left")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, unmet_mt, color=[RED, AMBER, GREEN])
    for xi, mt, pct in zip(x, unmet_mt, unmet_pct):
        ax2.text(xi, mt + 0.2, f"{mt:.1f} Mt\n({pct:.1f}%)", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Global unmet demand (Mt)")
    ax2.set_ylim(0, max(unmet_mt) * 1.25)
    ax2.set_title("Global unmet demand after the shock")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{subtitle} \u2014 global fertilizer impact",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = FIG / f"fig3_global_summary{_fig_suffix(tag)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scenario 1 publication figures.")
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"Scenario result tag (default: {DEFAULT_TAG})",
    )
    args = parser.parse_args()
    tag = args.tag
    subtitle = TAG_SUBTITLES.get(
        tag,
        f"Scenario tag: {tag}",
    )

    iso3 = build_iso3_map()
    fig_coverage_maps(iso3, tag, subtitle)
    fig_coverage_map_liebig(iso3, tag, subtitle)
    fig_major_countries(tag, subtitle)
    fig_global_summary(tag, subtitle)
    print("\nAll figures written to", FIG)
