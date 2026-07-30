# Scenario 1b — US/Russia Nuclear War with Reduced Trade

> Variant of scenario 1 that adds a **trade shock** on top of the same production
> shock. Implements the narrative that fertilizer logistics to/from the two
> belligerents would be as impaired as domestic production after a full-scale
> nuclear exchange.

---

## S.1 Scenario definition

Identical production shock to scenario 1; additionally, every bilateral flow in
the historical trade matrix `T0` where **either** the exporter **or** the
importer is the United States or Russian Federation is scaled by a surviving
fraction of **0.40** (−60 %).

| Parameter | United States / Russia | Rest of world |
|---|---|---|
| Surviving production fraction | 0.40 (−60 %) | 0.82 (−18 %) |
| Surviving trade fraction (flows touching US/RU) | 0.40 (−60 %) | unchanged |
| Demand | baseline (unchanged) | baseline (unchanged) |

The **baseline** RAS run uses unmodified `T0`. The **shocked** run uses
`P_shocked` and `T0_shocked = apply_trade_shock(T0, US/RU, 0.40)`.

Scenario tag: `scenario1_us_ru_reduced_trade`.

## S.2 Implementation

New helper in `src/preprocessing.py`:

- `apply_trade_shock(T0, countries, fraction)` — scale all cells touching
  any country in `countries`.
- `apply_trade_shock_reported(...)` — same, plus volume summary.

Notebook: `scripts/scenario1_us_russia_reduced_trade_all_nutrients.ipynb`.

## S.3 Results vs scenario 1 (production shock only)

With FAOSTAT 2014–2018 data, **per-country supply coverage is unchanged** at
one-decimal precision relative to scenario 1. Bilateral trade matrices `X`
*do* differ — partner routing shifts — but Phase 1 export/import targets
(`S_hat`, `D_hat`) are unchanged, and the RAS fitter rescales row/column
multipliers to match those targets regardless of the uniform `T0` scaling.

| Nutrient | Global coverage (scenario 1b) | US coverage | Russia coverage |
|---|---|---|---|
| Nitrogen (N) | 82.5 % | 67.0 % | 100.0 % |
| Phosphate (P₂O₅) | 77.0 % | 79.1 % | 100.0 % |
| Potash (K₂O) | 86.5 % | 83.4 % | 100.0 % |

These match scenario 1 exactly for coverage; trade-flow totals and partner
allocations differ.

## S.4 Interpretation

- **What the trade shock does in this model:** reduces the *historical weight*
  of US/RU-linked routes in the RAS structure matrix. The algorithm still
  converges to the same row/column totals dictated by post-shock production
  and baseline demand.
- **What it does not do (with a uniform surviving fraction):** cut total US/RU
  exports or imports below what Phase 1–2 already allow. For stronger
  isolation effects, consider:
  - `TRADE_SURVIVING_FRACTION = 0.0` (full embargo on US/RU flows)
  - export-only or import-only column/row zeroing
  - demand shock for devastated regions (not implemented)
- **Why keep scenario 1b:** documents the trade-shock machinery, enables
  sensitivity runs (embargo, asymmetric export/import cuts), and changes flow
  maps even when coverage is unchanged.

## S.5 Caveats

All scenario 1 caveats apply (fixed demand, uniform rest-of-world cut,
proportional-redistribution artefact). Additionally:

- Uniform `T0` scaling is **neutral for RAS totals** when the same factor
  applies to every cell touching a country; coverage may be identical to
  scenario 1 while `X` differs.
- For publication, compare flow maps / Sankey diagrams between scenario 1 and
  1b to show rerouting; do not expect coverage deltas from this specific
  parameterisation alone.

---

### Reproduction

```text
scripts/scenario1_us_russia_reduced_trade_all_nutrients.ipynb
```

Outputs:

- `results/summary_{N,P,K}_scenario1_us_ru_reduced_trade_{baseline,shocked}.csv`
- `results/cross_summary_scenario1_us_ru_reduced_trade.csv`
- `results/us_ru_coverage_compare_scenario1_us_ru_reduced_trade.csv` (if scenario 1 CSVs exist)

Figures: `python scripts/make_scenario1_figures.py --tag scenario1_us_ru_reduced_trade`

Output files use the `_scenario1_us_ru_reduced_trade` suffix (e.g.
`fig1_coverage_map_N_scenario1_us_ru_reduced_trade.png`).
