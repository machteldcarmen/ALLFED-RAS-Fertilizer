# Scenario 1 — US/Russia Nuclear War: Fertilizer Availability Results

> Draft results section for the forward-looking application of the four-phase
> RAS trade-redistribution model (validated against the 2008–2009 crisis in
> `paper_ras_validation.md`). All figures regenerated from FAOSTAT 2014–2018
> data after excluding regional/economic aggregates (Area Code ≥ 5000) and the
> redundant "China" roll-up (kept "China, mainland").

---

## S.1 Scenario definition

We simulate a US–Russia nuclear exchange and the ensuing global industrial
collapse (nuclear winter) as a one-off **production shock** applied to the
2014–2018 baseline, then redistribute the surviving supply with the RAS model.

| Country group | Surviving production fraction | Implied cut |
|---|---|---|
| United States of America | 0.40 | −60% |
| Russian Federation | 0.40 | −60% |
| Every other country | 0.82 | −18% |

The two heavy-hit producers are directly targeted; the rest-of-world −18%
represents the broader industrial/energy disruption of a nuclear winter. Demand
($C$) is held at its baseline level, so coverage changes reflect supply loss
and trade rerouting only.

## S.2 Magnitude of the production shock

**Table S.1.** Baseline production, post-shock production, and the role of the
two targeted producers.

| Nutrient | Countries | Baseline prod. (t) | Post-shock prod. (t) | Global decline | US+Russia share of baseline |
|---|---|---|---|---|---|
| Nitrogen (N) | 158 | 119,063,036 | 89,249,194 | −25.0% | 16.8% |
| Phosphate (P₂O₅) | 145 | 46,551,152 | 34,039,380 | −26.9% | 21.1% |
| Potash (K₂O) | 142 | 42,982,553 | 31,537,455 | −26.6% | 20.5% |

The baseline nitrogen total (119 Mt) matches independently reported global N
fertilizer production, confirming the aggregate-cleaning step. The combined
shock produces a 25–27% global production decline across nutrients, consistent
with the intended ~25% industrial-loss target.

## S.3 Global supply coverage after the shock

**Table S.2.** Global fertilizer coverage (availability ÷ demand) and resulting
unmet demand.

| Nutrient | Coverage baseline | Coverage post-shock | Change | Global unmet demand |
|---|---|---|---|---|
| Nitrogen (N) | 100% | 82.5% | −17.5 pp | 18.96 Mt (17.5% of demand) |
| Phosphate (P₂O₅) | 100% | 77.0% | −23.0 pp | 10.15 Mt (23.0% of demand) |
| Potash (K₂O) | 100% | 86.5% | −13.5 pp | 4.91 Mt (13.5% of demand) |

**Phosphate is the most disrupted nutrient** (23% global shortfall) and
**potash the least** (13.5%). The ordering follows production geography: potash
output is concentrated in Canada, Belarus and Russia, and Russia's large
domestic surplus is retained under the "domestic-first" filter, cushioning the
global pool less unevenly than for phosphate, whose supply is more import-reliant
worldwide.

## S.4 Who is hit hardest — the import-dependence divide

The defining result is a sharp split between self-sufficient producers and
import-dependent consumers.

**Pure importers cluster at a single coverage level.** Because RAS scales every
surviving trade flow proportionally, countries with zero domestic production all
converge to the same post-shock coverage:

| Nutrient | Import-only countries | Post-shock coverage |
|---|---|---|
| Nitrogen | 80 | 48.4% |
| Phosphate | 85 | 42.9% |
| Potash | 115 | 82.6% |

For nitrogen and phosphate this means **import-dependent countries lose more than
half their supply** (down to ~48% and ~43% of demand respectively). Most of
Sub-Saharan Africa, the Caribbean and many small/mid-sized states fall in this
group.

**Table S.3.** Post-shock coverage for major fertilizer consumers — Nitrogen.

| Country | Demand (t) | Coverage baseline | Coverage post-shock | Change |
|---|---|---|---|---|
| China (mainland) | 29,970,308 | 100% | 99.6% | −0.4 pp |
| India | 17,128,700 | 100% | 81.0% | −19.0 pp |
| United States | 11,749,885 | 100% | 67.0% | −33.0 pp |
| Brazil | 4,162,896 | 100% | 55.3% | −44.7 pp |
| Pakistan | 3,287,104 | 100% | 86.7% | −13.3 pp |
| Indonesia | 2,976,222 | 100% | 100.0% | 0.0 pp |
| Canada | 2,565,000 | 100% | 100.0% | 0.0 pp |
| France | 2,221,644 | 100% | 55.9% | −44.1 pp |
| Germany | 1,606,235 | 100% | 84.0% | −16.0 pp |
| Russia | 1,391,587 | 100% | 100.0% | 0.0 pp |
| Viet Nam | 1,548,934 | 100% | 76.1% | −23.9 pp |
| Bangladesh | 1,258,904 | 100% | 62.4% | −37.6 pp |
| Ethiopia | 292,071 | 100% | 48.4% | −51.6 pp |

Three patterns emerge across all three nutrients:

1. **Large self-sufficient producers are barely affected.** China remains near
   100% for N and P (it produces almost all it consumes); Russia stays at 100%
   for every nutrient because its retained domestic surplus exceeds its modest
   domestic demand even after a 60% cut.
2. **Targeted-but-exporting countries absorb the cut domestically.** The United
   States, despite a 60% production cut, falls only to 67% for N (it loses its
   export surplus first); the impact on US *consumption* is real but smaller
   than the headline production cut.
3. **Large importers are the most exposed major economies.** Brazil — the
   world's largest net importer of all three nutrients — drops to 55% (N), 62%
   (P) and 83% (K). France and several European importers fall by 30–47 pp.

Phosphate (Table-equivalent figures): Brazil 61.8%, Indonesia 63.7%, Pakistan
64.6%, India 74.1%, US 79.1%, while China and Russia remain at 100%. Potash is
milder for almost everyone (most importers at 82.6%), with China at 91.0%.

## S.5 Trade-volume response

Total modelled trade *falls* after the shock (N: −7.6 Mt; P: −6.6 Mt; K: −3.3 Mt
relative to the baseline RAS run). Under the RAS inertia assumption the surviving
export pool shrinks roughly in proportion to lost production, so trade contracts
rather than expands to compensate. This is the same conservative bias documented
in the validation study (`paper_ras_validation.md`, §4.5): real-world networks
would likely open new routes that the model cannot represent, so these coverage
figures should be read as a **plausible lower bound** on post-shock availability.

## S.6 Figures

Generated by `scripts/make_scenario1_figures.py` into `results/figures/`.

**Figure S.1 — Post-shock supply coverage per country (choropleth, one per nutrient).**
`fig1_coverage_map_N.png`, `fig1_coverage_map_P.png`, `fig1_coverage_map_K.png`.
Green = near self-sufficient, red = severe shortfall. The maps make the
import-dependence divide visible at a glance: large producers (China, Russia,
Canada) stay green while import-reliant regions (much of Sub-Saharan Africa,
parts of South America and South/South-East Asia) turn red.

**Figure S.2 — Post-shock coverage for major consumers, by nutrient (grouped bars).**
`fig2_major_countries.png`. Countries sorted by nitrogen coverage; the spread
from Russia/Canada/China (~100%) down to Ethiopia/Brazil/France (~50–55% for N)
illustrates exposure among the largest economies.

**Figure S.3 — Global impact summary.** `fig3_global_summary.png`. Left: global
coverage baseline vs post-shock; right: global unmet demand (Mt and % of demand)
per nutrient.

## S.7 Caveats specific to this scenario

- **Demand held fixed.** In a true nuclear-winter, agricultural demand would
  itself change (collapsing farm output, altered cropping); coverage here
  isolates the *supply* channel only.
- **Uniform rest-of-world cut.** The −18% is a single industrial-loss
  assumption; sensitivity to this parameter is not yet explored.
- **Proportional-redistribution artefact.** The clustering of all pure importers
  at one coverage value is a model property, not an empirical prediction; per-
  country precision for small importers is low (see validation paper §3.3).
- **Conservative by construction.** No new trade routes; results understate
  adaptive capacity.
- **Scenario 1b (reduced trade).** A variant with an additional −60 %
  trade-shock on all flows touching US/Russia is available as
  `scenario1_us_ru_reduced_trade` (see `docs/scenario1b_reduced_trade_results.md`).
  With uniform `T0` scaling, per-country coverage matches scenario 1; bilateral
  flow maps differ.

---

### Reproduction

Regenerate all CSVs by running `scripts/scenario1_us_russia_nuclear_all_nutrients.ipynb`
(now using the aggregate-cleaned `src/preprocessing.py`). Per-country summaries:
`results/summary_{N,P,K}_scenario1_us_ru_nuclear_{baseline,shocked}.csv`;
cross-nutrient table: `results/cross_summary_scenario1_us_ru_nuclear.csv`.
