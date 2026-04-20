"""Plotting helpers.

All plotting is kept here so the model / preprocessing / postprocessing
modules stay plot-free and therefore light to import.

Two backends are used:

* **plotly** for interactive figures (Sankey, heatmap, grouped bars).
  Returns a :class:`plotly.graph_objects.Figure` — the caller decides to
  ``.show()`` in a notebook, ``.write_html(...)`` to disk, etc.
* **matplotlib** for the baseline-vs-shocked comparison dashboard.
  Returns a :class:`matplotlib.figure.Figure`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import RASResult


ALLFED_MPLSTYLE_URL = (
    "https://raw.githubusercontent.com/allfed/"
    "ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


def use_allfed_style() -> bool:
    """Activate the ALLFED matplotlib style sheet.

    Tries the online stylesheet first; falls back silently to the default
    matplotlib style if offline or if the URL is unreachable, so tests
    and CI without internet don't break.

    Returns:
        bool: True if the ALLFED style was successfully applied.
    """
    import matplotlib.pyplot as plt

    try:
        plt.style.use(ALLFED_MPLSTYLE_URL)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Plotly: interactive single-run figures
# ──────────────────────────────────────────────────────────────────────────────
def plot_sankey(
    result: RASResult,
    title: str = "Post-RAS trade flows (X)",
    min_flow: float = 1e-9,
):
    """Sankey diagram of the bilateral trade matrix ``X``.

    Nodes are colored green if the country is a net exporter
    (``S_hat > 0``) and blue otherwise.
    """
    import plotly.graph_objects as go

    X = result.X
    labels = list(X.index)
    node_map = {c: i for i, c in enumerate(labels)}

    sources, targets, values = [], [], []
    for exporter in X.index:
        for importer in X.columns:
            v = float(X.loc[exporter, importer])
            if v > min_flow:
                sources.append(node_map[exporter])
                targets.append(node_map[importer])
                values.append(v)

    colors = ["#2ca02c" if result.S_hat[c] > 0 else "#1f77b4" for c in labels]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=22,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(100, 149, 237, 0.45)",
                    hovertemplate=(
                        "%{source.label} -> %{target.label}"
                        "<br>%{value:,.1f}<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(title=title, font_size=11, height=520)
    return fig


def plot_heatmap(
    result: RASResult,
    title: str = "RAS trade matrix heatmap (X)",
):
    """Heatmap of the bilateral trade matrix (exporter x importer)."""
    import plotly.express as px

    fig = px.imshow(
        result.X,
        text_auto=".1f",
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Importer", y="Exporter", color="Flow"),
        title=title,
    )
    fig.update_layout(height=520)
    return fig


def plot_country_bars(
    result: RASResult,
    title: str = "Country balance (Production / Demand / Availability)",
):
    """Grouped bars per country: ``P``, ``C``, ``F_final``, ``Unmet demand``."""
    import plotly.graph_objects as go

    unmet = (result.C - result.F).clip(lower=0)
    df = pd.DataFrame(
        {
            "Production P": result.P,
            "Demand C": result.C,
            "F_final": result.F,
            "Unmet demand": unmet,
        }
    ).reindex(result.X.index)

    fig = go.Figure()
    for col, color in [
        ("Production P", "#636EFA"),
        ("Demand C", "#EF553B"),
        ("F_final", "#00CC96"),
        ("Unmet demand", "#AB63FA"),
    ]:
        fig.add_trace(
            go.Bar(
                name=col,
                x=df.index.tolist(),
                y=df[col].tolist(),
                marker_color=color,
            )
        )

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Country",
        yaxis_title="Quantity",
        height=520,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib: baseline-vs-shocked dashboard
# ──────────────────────────────────────────────────────────────────────────────
def plot_comparison_dashboard(
    baseline: RASResult,
    shocked: RASResult,
    comparison: pd.DataFrame,
    nutrient_name: str = "Nutrient",
    top_k: int = 20,
    top_flows: int = 8,
):
    """2x2 matplotlib dashboard of baseline vs shocked results.

    Panels:
        1. Top-``top_k`` most affected countries (supply coverage, pre/post)
        2. Shocked countries — production pre/post
        3. Distribution of supply coverage (pre/post)
        4. Largest bilateral trade-flow changes
    """
    import matplotlib.pyplot as plt

    use_allfed_style()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel 1 — most affected (supply coverage)
    ax = axes[0, 0]
    top = comparison.nsmallest(top_k, "Change_pp")
    y = np.arange(len(top))
    ax.barh(
        y,
        top["Cover_base_%"],
        height=0.4,
        label="Pre-shock",
        color="#2980b9",
        alpha=0.85,
    )
    ax.barh(
        y + 0.4,
        top["Cover_shock_%"],
        height=0.4,
        label="Post-shock",
        color="#c0392b",
        alpha=0.85,
    )
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Supply coverage (%)")
    ax.set_title(f"{top_k} most affected countries — {nutrient_name}")
    ax.axvline(100, color="gray", ls="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel 2 — shocked countries: production pre vs post
    ax2 = axes[0, 1]
    shocked_rows = comparison[comparison["P_baseline"] != comparison["P_shocked"]]
    if not shocked_rows.empty:
        y2 = np.arange(len(shocked_rows))
        ax2.barh(
            y2,
            shocked_rows["P_baseline"],
            height=0.4,
            label="Pre-shock",
            color="#2980b9",
            alpha=0.85,
        )
        ax2.barh(
            y2 + 0.4,
            shocked_rows["P_shocked"],
            height=0.4,
            label="Post-shock",
            color="#c0392b",
            alpha=0.85,
        )
        ax2.set_yticks(y2 + 0.2)
        ax2.set_yticklabels(shocked_rows.index, fontsize=9)
        ax2.set_xlabel("Production")
        ax2.set_title("Shocked countries — production")
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No shocked countries", ha="center", va="center")
        ax2.axis("off")

    # Panel 3 — coverage histogram
    ax3 = axes[1, 0]
    ax3.hist(
        comparison["Cover_base_%"].clip(0, 250).dropna(),
        bins=30,
        alpha=0.6,
        label="Pre-shock",
        color="#2980b9",
    )
    ax3.hist(
        comparison["Cover_shock_%"].clip(0, 250).dropna(),
        bins=30,
        alpha=0.6,
        label="Post-shock",
        color="#c0392b",
    )
    ax3.set_xlabel("Supply coverage (%)")
    ax3.set_ylabel("Number of countries")
    ax3.set_title("Distribution of supply coverage")
    ax3.axvline(100, color="gray", ls="--", alpha=0.5)
    ax3.legend(fontsize=9)

    # Panel 4 — largest bilateral trade-flow changes
    ax4 = axes[1, 1]
    dX = shocked.X - baseline.X
    stacked = dX.stack()
    drops = stacked.nsmallest(top_flows)
    gains = stacked.nlargest(top_flows)
    all_ch = pd.concat([drops, gains])
    labels = [f"{e} -> {i}" for e, i in all_ch.index]
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in all_ch.values]
    ax4.barh(np.arange(len(all_ch)), all_ch.values, color=colors, alpha=0.85)
    ax4.set_yticks(np.arange(len(all_ch)))
    ax4.set_yticklabels(labels, fontsize=7)
    ax4.set_xlabel("Trade flow change")
    ax4.set_title("Largest bilateral trade changes (red=down, green=up)")
    ax4.axvline(0, color="gray", ls="-", alpha=0.3)

    fig.suptitle(
        f"RAS model — {nutrient_name} — baseline vs shocked",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig
