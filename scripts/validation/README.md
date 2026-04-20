# Validation notebooks

These notebooks apply the same RAS model implemented in `src/` to
historical shock events, to test whether the model reproduces the
observed redistribution of fertilizer flows.

| Notebook                                      | Scenario                                                                                |
|-----------------------------------------------|-----------------------------------------------------------------------------------------|
| `fertilizer_validation_2009_crisis.ipynb`     | 2007-2009 fertilizer-price / financial-crisis shock on global N, P, K flows             |
| `fertilizer_validation_russia_ukraine.ipynb`  | 2022 Russia / Ukraine export disruption as a secondary case                             |

Both notebooks are **thin drivers**: they load FAOSTAT data via
`src.preprocessing`, apply a historical shock, run
`FertilizerRAS(...).run_shocked(...)` and compare the modelled
redistribution to the actually observed trade flows.

> The `_2009_crisis` notebook is ~2 MB because it still contains cell
> outputs. If you want to commit it slimmer, clear outputs first:
> `jupyter nbconvert --clear-output --inplace fertilizer_validation_2009_crisis.ipynb`.

For the underlying equations see `../../docs/methodology.md`.
