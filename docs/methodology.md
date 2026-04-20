# Methodology

This document describes the mathematical model implemented in `src/`. It
is written so that you can read the README first for the big picture,
then come here to understand exactly *which equations are solved* and
*where in the code they live*.

## 1. Symbols

Indices $i, j$ run over the set $N$ of countries (one set per nutrient).

| Symbol            | Meaning                                                    | Unit   |
|-------------------|------------------------------------------------------------|--------|
| $P_i$             | Post-shock production in country $i$                        | tonnes |
| $C_i$             | Domestic demand ("AgUse") in country $i$                    | tonnes |
| $T^0_{ij}$        | Historical (pre-shock) bilateral trade from $i$ to $j$      | tonnes |
| $B_i$             | Net balance $B_i = P_i - C_i$                               | tonnes |
| $K_i$             | Production kept for domestic use                            | tonnes |
| $S^*_i$           | Target export quantity (before global consistency)          | tonnes |
| $D^*_i$           | Target import quantity (before global consistency)          | tonnes |
| $\hat S_i, \hat D_j$ | Feasible export / import targets (after Phase 2)          | tonnes |
| $x_{ij}$          | Post-shock bilateral trade from $i$ to $j$ (model output)   | tonnes |
| $F_i$             | Final fertilizer availability in country $i$                | tonnes |
| $r_i, c_j$        | RAS row / column multipliers                                | –      |

The model runs **once per nutrient**; nutrients do not interact.

## 2. Phase 1 — Domestic first filter

Each country first tries to satisfy its own demand with its own
production; only the leftover is available for trade.

$$
B_i \;=\; P_i - C_i
$$

$$
(K_i, S^*_i, D^*_i) \;=\;
\begin{cases}
(\;C_i,\; B_i,\; 0\;)        & \text{if } B_i > 0 \quad \text{(surplus)} \\[2pt]
(\;P_i,\; 0,\; |B_i|\;)      & \text{if } B_i \le 0 \quad \text{(deficit)}
\end{cases}
\qquad \text{(eqs. 1–6)}
$$

**Code:** `FertilizerRAS._phase1` in `src/model.py`.

## 3. Phase 2 — Global consistency

RAS only has a solution when total feasible supply equals total feasible
demand:

$$
\sum_i \hat S_i \;=\; \sum_j \hat D_j.
$$

Let $S_{\text{tot}} = \sum_i S^*_i$ and $D_{\text{tot}} = \sum_i D^*_i$.

* **Global shortage** ($S_{\text{tot}} < D_{\text{tot}}$): supply is
  binding; scale demand down.

$$
\hat S_i = S^*_i, \qquad \hat D_j = D^*_j \cdot \frac{S_{\text{tot}}}{D_{\text{tot}}}
\qquad \text{(eqs. 7–8)}
$$

* **Global surplus** ($S_{\text{tot}} > D_{\text{tot}}$): demand is
  binding; scale supply down.

$$
\hat S_i = S^*_i \cdot \frac{D_{\text{tot}}}{S_{\text{tot}}}, \qquad \hat D_j = D^*_j
\qquad \text{(eqs. 9–10)}
$$

* **Equal totals:** $\hat S_i = S^*_i$, $\hat D_j = D^*_j$.

* **Degenerate ($S_{\text{tot}} = 0$ or $D_{\text{tot}} = 0$):** no trade
  is possible; we set $\hat S = \hat D = 0$, which in turn makes $X = 0$
  in Phase 3. Phase 4 then returns $F_i = K_i$.

**Code:** `FertilizerRAS._phase2` in `src/model.py`.

## 4. Phase 3 — RAS / iterative proportional fitting

We look for a matrix $X$ with the same zero-pattern and approximately
the same structure as $T^0$, whose row sums match the export targets
$\hat S$ and whose column sums match the import targets $\hat D$.

The solution has the bi-proportional form

$$
x_{ij} \;=\; r_i \cdot T^0_{ij} \cdot c_j
\qquad \text{(eq. 11)}
$$

with $r_i, c_j > 0$ found by alternately enforcing the row and column
constraints:

$$
r_i \;\leftarrow\; r_i \cdot \frac{\hat S_i}{\sum_j r_i\, T^0_{ij}\, c_j}
\qquad \text{(eq. 12)}
$$

$$
c_j \;\leftarrow\; c_j \cdot \frac{\hat D_j}{\sum_i r_i\, T^0_{ij}\, c_j}
\qquad \text{(eq. 13)}
$$

Convergence criterion:

$$
\max\!\Big(\,
\max_i |{\textstyle\sum_j} x_{ij} - \hat S_i|,\;
\max_j |{\textstyle\sum_i} x_{ij} - \hat D_j|
\,\Big) < \varepsilon.
$$

We use $\varepsilon = 10^{-6}$ by default and cap iterations at 1000.

**Numerical safety.** If a row sum is exactly 0 (the country has no
historical trade partners, so $T^0_{i,\cdot} = 0$), we keep its
multiplier $r_i$ unchanged instead of dividing by zero. The same holds
for columns.

**Code:** `run_ras` in `src/model.py`.

## 5. Phase 4 — Final availability

$$
F_i \;=\; K_i + \sum_{j \in N} x_{ji}
\qquad \text{(eq. 14)}
$$

By construction $\sum_i F_i = \sum_i K_i + \sum_{i,j} x_{ji}
= \sum_i K_i + \sum_j \hat S_j \approx \sum_i (K_i + S^*_i) = \sum_i P_i$
up to the Phase-2 rescaling. We report per-country *supply coverage*
$F_i / C_i \cdot 100\%$ and *unmet demand* $\max(0,\; C_i - F_i)$.

**Code:** `FertilizerRAS.run` in `src/model.py`.

## 6. Baseline vs shocked run

A *shock* is a dict `{country: surviving_production_fraction}`.
`FertilizerRAS.run_shocked(shock)` runs the full pipeline twice: once
with the unmodified production vector (the *baseline*), and once with
the shocked one, returning both results. The post-processing module
then produces a side-by-side comparison (see `src/postprocessing.py`
and the `scripts/real_data_nitrogen.ipynb` notebook).

## 7. Assumptions and limitations

* **Re-exports are not corrected.** The trade matrix $T^0$ is taken at
  face value from the FAOSTAT *Detailed Trade Matrix* (exports).
  Countries that act as trade hubs may be over-represented. If
  re-export correction becomes important, see the Croft et al. (2018)
  algorithm as implemented in
  [`allfed/pytradeshifts`](https://github.com/allfed/pytradeshifts).
* **Nutrients are treated independently.** There is no substitution
  between N, P, and K in the model.
* **No gravity penalty.** Every historical trade edge is treated equally
  regardless of distance. A long-distance trade penalty in the style of
  `pytradeshifts` could be added by multiplying $T^0_{ij}$ by a distance
  kernel before Phase 3.
* **Small countries are filtered out.** We drop countries whose
  production *and* demand are both below `min_threshold` (default
  1000 t). This is configurable in `src/preprocessing.filter_countries`.
* **Units.** The model is unit-agnostic: as long as $P$, $C$, and $T^0$
  share the same unit, the outputs are in that unit.

## 8. References

* Stone, R. (1961). *Input-Output and National Accounts*. OEEC, Paris.
  (Bi-proportional fitting / RAS.)
* FAOSTAT: Food and Agriculture Organization of the United Nations.
  https://www.fao.org/faostat/
* Croft, S. A., West, C. D., & Green, J. M. H. (2018). Capturing the
  heterogeneity of sub-national production in global trade flows.
  *Journal of Cleaner Production*, **203**, 1106–1118.
  https://doi.org/10.1016/j.jclepro.2018.08.267
* ALLFED, *ALLFED Task for math students – modelling trade* (internal
  brief, 2026).
