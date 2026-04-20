# scripts

Thin Jupyter notebooks that drive the package in `../src/`. They
contain **no business logic** — only imports, calls and visualisations.

| Notebook                     | What it shows                                                   |
|------------------------------|-----------------------------------------------------------------|
| `toy_example.ipynb`          | 5-country demo, no external data needed                         |
| `real_data_nitrogen.ipynb`   | Full FAOSTAT pipeline for nitrogen, with a production shock     |
| `all_nutrients.ipynb`        | Same pipeline repeated for N, P and K                           |
| `validation/*.ipynb`         | Historical-shock validation (2009 fertilizer crisis, 2022 Russia-Ukraine) |

The notebooks expect the FAOSTAT CSVs under `../data/`
(see `../data/README.md`).
