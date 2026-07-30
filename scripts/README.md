# scripts

Thin Jupyter notebooks that drive the package in `../src/`. They
contain **no business logic** — only imports, calls and visualisations.

| Notebook                     | What it shows                                                   |
|------------------------------|-----------------------------------------------------------------|
| `toy_example.ipynb`          | 5-country demo, no external data needed                         |
| `real_data_nitrogen.ipynb`   | Full FAOSTAT pipeline for nitrogen, with a production shock     |
| `scenario1_us_russia_nuclear.ipynb` | Scenario 1 (US/RU −60 %, world −18 %) for one nutrient (N by default) |
| `scenario1_us_russia_nuclear_all_nutrients.ipynb` | Same scenario 1 for N, P and K in one loop |
| `scenario1_us_russia_reduced_trade_all_nutrients.ipynb` | Scenario 1b: same production shock + −60 % trade on US/RU flows (N, P, K) |
| `all_nutrients.ipynb`        | N, P, K with a Russia/Belarus/Ukraine-only shock (not scenario 1) |
| `validation/*.ipynb`         | Historical-shock validation (2009 fertilizer crisis, 2022 Russia-Ukraine) |

The notebooks expect the FAOSTAT CSVs under `../data/`
(see `../data/README.md`).
