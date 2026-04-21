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
# Country metadata for geographic plots
# ──────────────────────────────────────────────────────────────────────────────
# Centroid (lat, lon), ISO-3 code and plotly-friendly display name for a
# selection of common countries. Used by the choropleth / network helpers
# below. Users can pass ``country_meta=...`` to extend or override this.
_DEFAULT_COUNTRY_META: dict[str, dict] = {
    "Russia":     {"iso3": "RUS", "lat":  61.5, "lon":  105.3, "display": "Russia"},
    "Russian Federation": {"iso3": "RUS", "lat": 61.5, "lon": 105.3, "display": "Russia"},
    "China":      {"iso3": "CHN", "lat":  35.0, "lon":  103.8, "display": "China"},
    "China, mainland": {"iso3": "CHN", "lat": 35.0, "lon": 103.8, "display": "China"},
    "USA":        {"iso3": "USA", "lat":  39.8, "lon":  -98.5, "display": "United States"},
    "United States": {"iso3": "USA", "lat": 39.8, "lon": -98.5, "display": "United States"},
    "United States of America": {"iso3": "USA", "lat": 39.8, "lon": -98.5, "display": "United States"},
    "India":      {"iso3": "IND", "lat":  21.0, "lon":   78.0, "display": "India"},
    "Brazil":     {"iso3": "BRA", "lat": -10.0, "lon":  -55.0, "display": "Brazil"},
    "Canada":     {"iso3": "CAN", "lat":  56.0, "lon": -106.0, "display": "Canada"},
    "Mexico":     {"iso3": "MEX", "lat":  23.6, "lon": -102.5, "display": "Mexico"},
    "Argentina":  {"iso3": "ARG", "lat": -38.4, "lon":  -63.6, "display": "Argentina"},
    "Chile":      {"iso3": "CHL", "lat": -35.7, "lon":  -71.5, "display": "Chile"},
    "Germany":    {"iso3": "DEU", "lat":  51.2, "lon":   10.5, "display": "Germany"},
    "France":     {"iso3": "FRA", "lat":  46.2, "lon":    2.2, "display": "France"},
    "United Kingdom": {"iso3": "GBR", "lat": 54.0, "lon": -2.5, "display": "United Kingdom"},
    "UK":         {"iso3": "GBR", "lat":  54.0, "lon":   -2.5, "display": "United Kingdom"},
    "Italy":      {"iso3": "ITA", "lat":  41.9, "lon":   12.6, "display": "Italy"},
    "Spain":      {"iso3": "ESP", "lat":  40.5, "lon":   -3.7, "display": "Spain"},
    "Poland":     {"iso3": "POL", "lat":  51.9, "lon":   19.1, "display": "Poland"},
    "Netherlands": {"iso3": "NLD", "lat":  52.1, "lon":    5.3, "display": "Netherlands"},
    "Belarus":    {"iso3": "BLR", "lat":  53.7, "lon":   27.9, "display": "Belarus"},
    "Ukraine":    {"iso3": "UKR", "lat":  48.4, "lon":   31.2, "display": "Ukraine"},
    "Turkey":     {"iso3": "TUR", "lat":  38.9, "lon":   35.2, "display": "Turkey"},
    "Egypt":      {"iso3": "EGY", "lat":  26.8, "lon":   30.8, "display": "Egypt"},
    "Morocco":    {"iso3": "MAR", "lat":  31.8, "lon":   -7.1, "display": "Morocco"},
    "Nigeria":    {"iso3": "NGA", "lat":   9.1, "lon":    8.7, "display": "Nigeria"},
    "South Africa": {"iso3": "ZAF", "lat": -30.6, "lon":  22.9, "display": "South Africa"},
    "Saudi Arabia": {"iso3": "SAU", "lat": 23.9, "lon":   45.1, "display": "Saudi Arabia"},
    "Iran":       {"iso3": "IRN", "lat":  32.4, "lon":   53.7, "display": "Iran"},
    "Iran (Islamic Republic of)": {"iso3": "IRN", "lat": 32.4, "lon": 53.7, "display": "Iran"},
    "Pakistan":   {"iso3": "PAK", "lat":  30.4, "lon":   69.3, "display": "Pakistan"},
    "Bangladesh": {"iso3": "BGD", "lat":  23.7, "lon":   90.4, "display": "Bangladesh"},
    "Indonesia":  {"iso3": "IDN", "lat":  -0.8, "lon":  113.9, "display": "Indonesia"},
    "Vietnam":    {"iso3": "VNM", "lat":  14.1, "lon":  108.3, "display": "Viet Nam"},
    "Viet Nam":   {"iso3": "VNM", "lat":  14.1, "lon":  108.3, "display": "Viet Nam"},
    "Thailand":   {"iso3": "THA", "lat":  15.9, "lon":  101.0, "display": "Thailand"},
    "Japan":      {"iso3": "JPN", "lat":  36.2, "lon":  138.3, "display": "Japan"},
    "Republic of Korea": {"iso3": "KOR", "lat": 35.9, "lon": 127.8, "display": "South Korea"},
    "South Korea": {"iso3": "KOR", "lat": 35.9, "lon":  127.8, "display": "South Korea"},
    "Australia":  {"iso3": "AUS", "lat": -25.3, "lon":  133.8, "display": "Australia"},
    "New Zealand": {"iso3": "NZL", "lat": -40.9, "lon": 174.9, "display": "New Zealand"},
}


def _resolve_meta(country: str, country_meta: dict | None) -> dict | None:
    """Return the metadata entry for ``country`` or ``None`` if unknown."""
    if country_meta and country in country_meta:
        return country_meta[country]
    return _DEFAULT_COUNTRY_META.get(country)


# ──────────────────────────────────────────────────────────────────────────────
# Plotly: choropleth maps (production / demand)
# ──────────────────────────────────────────────────────────────────────────────
def _plot_choropleth(
    values: pd.Series,
    title: str,
    colorscale: str,
    colorbar_title: str,
    country_meta: dict | None = None,
):
    """Shared choropleth helper; maps a per-country Series onto a world map."""
    import plotly.graph_objects as go

    locations, z, hover_text = [], [], []
    missing = []
    for country, val in values.items():
        meta = _resolve_meta(country, country_meta)
        if meta is None or "iso3" not in meta:
            missing.append(country)
            continue
        locations.append(meta["iso3"])
        z.append(float(val))
        hover_text.append(
            f"<b>{meta.get('display', country)}</b><br>{colorbar_title}: {val:,.1f}"
        )

    fig = go.Figure(
        data=go.Choropleth(
            locations=locations,
            z=z,
            locationmode="ISO-3",
            colorscale=colorscale,
            marker_line_color="white",
            marker_line_width=0.5,
            colorbar_title=colorbar_title,
            hovertext=hover_text,
            hoverinfo="text",
        )
    )

    subtitle = ""
    if missing:
        subtitle = (
            f"<br><sub>(no ISO mapping for: {', '.join(missing[:8])}"
            f"{'…' if len(missing) > 8 else ''})</sub>"
        )

    fig.update_layout(
        title=title + subtitle,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
        ),
        height=520,
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def plot_production_map(
    P: pd.Series,
    title: str = "Post-shock production per country",
    country_meta: dict | None = None,
    colorscale: str = "Greens",
):
    """Choropleth world map of (post-shock) production ``P``.

    Parameters
    ----------
    P
        Country-indexed production series.
    title
        Figure title.
    country_meta
        Optional mapping ``{country: {"iso3": ..., "lat": ..., "lon": ...,
        "display": ...}}`` that extends / overrides the built-in default.
    colorscale
        Any plotly continuous colorscale name.
    """
    return _plot_choropleth(
        P, title=title, colorscale=colorscale,
        colorbar_title="Production", country_meta=country_meta,
    )


def plot_demand_map(
    C: pd.Series,
    title: str = "Post-shock demand per country",
    country_meta: dict | None = None,
    colorscale: str = "Reds",
):
    """Choropleth world map of (post-shock) demand ``C``."""
    return _plot_choropleth(
        C, title=title, colorscale=colorscale,
        colorbar_title="Demand", country_meta=country_meta,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plotly: chord diagram (historical trade matrix)
# ──────────────────────────────────────────────────────────────────────────────
# Palette mirrors the d3 chord diagram in
# ``Allfed/allfed-fertilizer-ras-main/fertilizer_trade_visualizations*.html``.
_CHORD_PALETTE = [
    "#c0392b", "#e67e22", "#2980b9", "#27ae60", "#8e44ad",
    "#16a085", "#d35400", "#2c3e50", "#f39c12", "#1abc9c",
    "#e74c3c", "#3498db", "#9b59b6", "#34495e",
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` + alpha in [0, 1] to a plotly ``rgba(...)`` string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _arc_xy(theta0: float, theta1: float, radius: float, n_pts: int = 32):
    """Polar-sampled arc from ``theta0`` to ``theta1`` at ``radius``."""
    theta = np.linspace(theta0, theta1, n_pts)
    return radius * np.cos(theta), radius * np.sin(theta)


def _quad_bezier_xy(p0, p1, ctrl=(0.0, 0.0), n_pts: int = 30):
    """Quadratic Bézier points from ``p0`` to ``p1`` through ``ctrl``."""
    t = np.linspace(0, 1, n_pts)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p1[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p1[1]
    return x, y


def plot_chord_diagram(
    T: pd.DataFrame,
    title: str = "Historical trade matrix — chord diagram",
    min_flow: float = 1e-9,
    palette: list[str] | None = None,
):
    """d3-style chord diagram of a bilateral trade matrix.

    Mirrors the visual convention of ``d3.chord()`` used in the ALLFED
    fertilizer-trade HTML dashboards:

    * each country is a filled outer arc whose angular size is proportional
      to its total trade (row + column sum in ``T``),
    * each non-zero flow ``T[i, j]`` is a filled ribbon that connects a
      sub-arc on country ``i`` (width ∝ T[i, j]) to a sub-arc on country
      ``j`` (same width), drawn as two inner-radius arcs joined by two
      quadratic Béziers through the circle's origin,
    * sub-arcs within each country are sorted by descending magnitude,
    * ribbons are colored by their source (exporter) and rendered at 45 %
      opacity so overlapping flows read clearly,
    * labels sit just outside the outer arcs.

    Implemented with pure plotly primitives — no extra dependencies.

    Parameters
    ----------
    T
        Square bilateral flow matrix (exporter × importer). ``T.loc[i, j]``
        is the flow from ``i`` to ``j``. The diagonal is ignored.
    title
        Figure title.
    min_flow
        Flows ``<= min_flow`` are omitted.
    palette
        Optional list of colors to cycle through for countries. Defaults to
        the d3 chord palette used in the HTML dashboards.
    """
    import plotly.graph_objects as go

    if not T.index.equals(T.columns):
        raise ValueError("T must be square with identical index and columns.")

    countries = list(T.index)
    n = len(countries)
    if n == 0:
        raise ValueError("T is empty.")

    M = T.to_numpy(dtype=float).copy()
    np.fill_diagonal(M, 0.0)
    M[M < 0] = 0.0

    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    totals = row_sums + col_sums
    total = float(totals.sum())
    if total <= 0:
        raise ValueError("T has no non-zero flows.")

    colors = palette if palette else _CHORD_PALETTE
    node_color = [colors[i % len(colors)] for i in range(n)]

    pad_angle = 0.02 if n > 15 else 0.04
    usable = 2 * np.pi - n * pad_angle
    if usable <= 0:
        raise ValueError("Too many countries for the chord layout.")
    scale = usable / total

    starts = np.zeros(n)
    ends = np.zeros(n)
    cursor = 0.0
    for i in range(n):
        starts[i] = cursor
        ends[i] = cursor + totals[i] * scale
        cursor = ends[i] + pad_angle

    out_sub: list[list[tuple[float, float] | None]] = [[None] * n for _ in range(n)]
    in_sub: list[list[tuple[float, float] | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        subs: list[tuple[str, int, float]] = []
        for j in range(n):
            if i == j:
                continue
            if M[i, j] > 0:
                subs.append(("out", j, float(M[i, j])))
            if M[j, i] > 0:
                subs.append(("in", j, float(M[j, i])))
        subs.sort(key=lambda x: -x[2])
        a = starts[i]
        for kind, j, v in subs:
            w = v * scale
            if kind == "out":
                out_sub[i][j] = (a, a + w)
            else:
                in_sub[i][j] = (a, a + w)
            a += w

    outer_r = 1.0
    inner_r = 0.88

    fig = go.Figure()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = float(M[i, j])
            if v <= min_flow:
                continue
            if out_sub[i][j] is None or in_sub[j][i] is None:
                continue

            theta_a, theta_b = out_sub[i][j]
            phi_a, phi_b = in_sub[j][i]

            xa, ya = _arc_xy(theta_a, theta_b, inner_r, n_pts=24)
            xc, yc = _arc_xy(phi_a, phi_b, inner_r, n_pts=24)
            p_tb = (inner_r * np.cos(theta_b), inner_r * np.sin(theta_b))
            p_pa = (inner_r * np.cos(phi_a), inner_r * np.sin(phi_a))
            p_pb = (inner_r * np.cos(phi_b), inner_r * np.sin(phi_b))
            p_ta = (inner_r * np.cos(theta_a), inner_r * np.sin(theta_a))
            xb1, yb1 = _quad_bezier_xy(p_tb, p_pa, n_pts=28)
            xb2, yb2 = _quad_bezier_xy(p_pb, p_ta, n_pts=28)

            xs_poly = np.concatenate([xa, xb1, xc, xb2])
            ys_poly = np.concatenate([ya, yb1, yc, yb2])

            fig.add_trace(
                go.Scatter(
                    x=xs_poly, y=ys_poly,
                    mode="lines",
                    line=dict(width=0.5, color=node_color[i]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(node_color[i], 0.45),
                    hoveron="fills",
                    hoverinfo="text",
                    text=f"<b>{countries[i]}</b> → <b>{countries[j]}</b><br>{v:,.1f}",
                    showlegend=False,
                )
            )

    for i in range(n):
        x_outer, y_outer = _arc_xy(starts[i], ends[i], outer_r, n_pts=60)
        x_inner, y_inner = _arc_xy(ends[i], starts[i], inner_r, n_pts=60)
        xs_poly = np.concatenate([x_outer, x_inner])
        ys_poly = np.concatenate([y_outer, y_inner])
        fig.add_trace(
            go.Scatter(
                x=xs_poly, y=ys_poly,
                mode="lines",
                line=dict(width=0.5, color=node_color[i]),
                fill="toself",
                fillcolor=node_color[i],
                hoveron="fills",
                hoverinfo="text",
                text=(
                    f"<b>{countries[i]}</b><br>"
                    f"Exports: {row_sums[i]:,.1f}<br>"
                    f"Imports: {col_sums[i]:,.1f}"
                ),
                showlegend=False,
            )
        )

    mid_angles = (starts + ends) / 2
    label_r = outer_r + 0.08
    annotations = []
    for i, angle in enumerate(mid_angles):
        deg = np.degrees(angle)
        flipped = np.cos(angle) < 0
        text_angle = -deg if not flipped else 180 - deg
        annotations.append(
            dict(
                x=label_r * np.cos(angle),
                y=label_r * np.sin(angle),
                text=countries[i],
                showarrow=False,
                xanchor="left" if not flipped else "right",
                yanchor="middle",
                textangle=text_angle,
                font=dict(size=11 if n > 15 else 13, color="#3d3d3a"),
            )
        )

    pad_xy = 0.4
    fig.update_layout(
        title=title,
        xaxis=dict(
            visible=False, range=[-1 - pad_xy, 1 + pad_xy],
            scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(visible=False, range=[-1 - pad_xy, 1 + pad_xy]),
        plot_bgcolor="#fafaf8",
        paper_bgcolor="#fafaf8",
        height=620,
        margin=dict(l=10, r=10, t=60, b=10),
        annotations=annotations,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plotly: geographic trade-network map
# ──────────────────────────────────────────────────────────────────────────────
def _geo_arc_lonlat(lon1, lat1, lon2, lat2, n_pts: int = 30):
    """Curved arc between two lon/lat points (quadratic Bézier, perpendicular offset).

    Mirrors the ``geoArc`` helper in the d3 HTML dashboard: the control point
    sits perpendicular to the chord at 22 % of its length, which gives the
    characteristic gently-bowed arc.
    """
    dx, dy = lon2 - lon1, lat2 - lat1
    d = np.sqrt(dx * dx + dy * dy)
    if d < 1e-6:
        return np.array([lon1, lon2]), np.array([lat1, lat2])
    bow = 0.22 * d
    nx, ny = -dy / d, dx / d
    mx = (lon1 + lon2) / 2 + nx * bow
    my = (lat1 + lat2) / 2 + ny * bow
    return _quad_bezier_xy((lon1, lat1), (lon2, lat2), ctrl=(mx, my), n_pts=n_pts)


def plot_trade_network_map(
    T: pd.DataFrame,
    title: str = "Trade network on world map",
    min_flow: float = 1e-9,
    country_meta: dict | None = None,
    tier_thresholds: tuple[float, float] | None = None,
    production: pd.Series | None = None,
):
    """Geographic trade network in the style of the ALLFED d3 HTML dashboards.

    * Natural-Earth world map with soft land / white country borders.
    * Each flow ``T[i, j]`` is a curved arc (quadratic Bézier with
      perpendicular offset) from exporter to importer.
    * Arcs are colored in three tiers (red / gold / blue) matching the
      d3 dashboard convention:
        - large  ≥ ``tier_thresholds[1]``  → ``#c0392b`` (red)
        - medium ≥ ``tier_thresholds[0]``  → ``#d4a017`` (gold)
        - small  < ``tier_thresholds[0]``  → ``#2980b9`` (blue)
      Opacity and width scale with the tier.
    * If ``tier_thresholds`` is ``None``, the 33rd / 67th percentiles of the
      non-zero flows are used so the same visual convention works for any
      data scale (e.g. the toy 5-country example with flows ~50–1 500 as
      well as real FAOSTAT data in tonnes).
    * Country markers use an RdYlBu sequential log color scale on
      production (or total throughput if ``production`` is ``None``) and
      ``sqrt``-scaled sizes.

    Parameters
    ----------
    T
        Square bilateral flow matrix (exporter × importer).
    title
        Figure title.
    min_flow
        Flows ``<= min_flow`` are omitted.
    country_meta
        Optional ``{country: {"iso3", "lat", "lon", "display"}}`` mapping
        extending / overriding the built-in default.
    tier_thresholds
        ``(mid, large)`` absolute cut-offs in the units of ``T``. When
        ``None``, adaptive 33 %/67 % percentiles of non-zero flows are used.
    production
        Optional country-indexed Series used for node sizing / coloring.
        Falls back to total throughput (row + column sums of ``T``).
    """
    import plotly.graph_objects as go

    if not T.index.equals(T.columns):
        raise ValueError("T must be square with identical index and columns.")

    countries = list(T.index)
    M = T.to_numpy(dtype=float).copy()
    np.fill_diagonal(M, 0.0)

    flows: list[tuple[str, str, float, dict, dict]] = []
    missing: list[str] = []
    for i, exporter in enumerate(countries):
        m_i = _resolve_meta(exporter, country_meta)
        if m_i is None:
            missing.append(exporter)
            continue
        for j, importer in enumerate(countries):
            if i == j:
                continue
            v = float(M[i, j])
            if v <= min_flow:
                continue
            m_j = _resolve_meta(importer, country_meta)
            if m_j is None:
                continue
            flows.append((exporter, importer, v, m_i, m_j))

    if tier_thresholds is not None:
        tier_mid, tier_large = tier_thresholds
    elif flows:
        values = np.array([f[2] for f in flows])
        tier_mid = float(np.percentile(values, 33.3))
        tier_large = float(np.percentile(values, 66.7))
    else:
        tier_mid, tier_large = 1.0, 2.0

    def tier_color(v: float) -> str:
        if v >= tier_large:
            return "#c0392b"
        if v >= tier_mid:
            return "#d4a017"
        return "#2980b9"

    def tier_opacity(v: float) -> float:
        if v >= tier_large:
            return 0.55
        if v >= tier_mid:
            return 0.40
        return 0.25

    max_v = max((f[2] for f in flows), default=1.0)
    log_max = np.log10(max_v + 1.0)

    def tier_width(v: float) -> float:
        if log_max <= 0:
            return 1.0
        return 0.5 + 3.5 * (np.log10(v + 1.0) / log_max)

    fig = go.Figure()

    for exporter, importer, v, m_i, m_j in flows:
        lons, lats = _geo_arc_lonlat(
            m_i["lon"], m_i["lat"], m_j["lon"], m_j["lat"], n_pts=30,
        )
        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=tier_width(v), color=tier_color(v)),
                opacity=tier_opacity(v),
                hoverinfo="text",
                text=f"{exporter} → {importer}: {v:,.1f}",
                showlegend=False,
            )
        )

    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    throughput = row_sums + col_sums

    if production is not None:
        prod_vals = production.reindex(countries).fillna(0).to_numpy(dtype=float)
        size_driver = prod_vals
        color_driver = prod_vals
        size_label = "Production"
    else:
        size_driver = throughput
        color_driver = throughput
        size_label = "Throughput"

    max_driver = float(np.max(size_driver)) if np.max(size_driver) > 0 else 1.0

    node_lon, node_lat, node_size, node_text, node_display, node_color = (
        [], [], [], [], [], []
    )
    for idx, country in enumerate(countries):
        meta = _resolve_meta(country, country_meta)
        if meta is None:
            continue
        node_lon.append(meta["lon"])
        node_lat.append(meta["lat"])
        val = float(size_driver[idx])
        size = 4.0 + 16.0 * np.sqrt(max(val, 0.0) / max_driver) if max_driver > 0 else 6.0
        node_size.append(size)
        node_color.append(float(color_driver[idx]))
        node_display.append(meta.get("display", country))
        node_text.append(
            f"<b>{meta.get('display', country)}</b>"
            f"<br>{size_label}: {val:,.1f}"
            f"<br>Exports: {row_sums[idx]:,.1f}"
            f"<br>Imports: {col_sums[idx]:,.1f}"
        )

    fig.add_trace(
        go.Scattergeo(
            lon=node_lon,
            lat=node_lat,
            mode="markers+text",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="RdYlBu",
                reversescale=True,
                cmin=0,
                cmax=max_driver,
                line=dict(color="rgba(0,0,0,0.35)", width=0.5),
                opacity=0.9,
                showscale=False,
            ),
            text=node_display,
            textposition="top center",
            textfont=dict(size=11, color="#2c2c2a"),
            hoverinfo="text",
            hovertext=node_text,
            showlegend=False,
        )
    )

    subtitle = ""
    if missing:
        subtitle = (
            f"<br><sub>(no coordinates for: {', '.join(missing[:8])}"
            f"{'…' if len(missing) > 8 else ''})</sub>"
        )

    fig.update_layout(
        title=title + subtitle,
        geo=dict(
            showland=True,
            landcolor="#e8e6df",
            showcountries=True,
            countrycolor="rgba(255,255,255,0.7)",
            coastlinecolor="#a0a09a",
            coastlinewidth=0.5,
            showframe=False,
            projection_type="natural earth",
            bgcolor="#fafaf8",
        ),
        paper_bgcolor="#fafaf8",
        height=540,
        margin=dict(l=10, r=10, t=70, b=10),
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
