# src

All core Python code for the package lives here. Nothing in this folder
uses a notebook; notebooks belong in `../scripts/`.

| Module              | What it does                                                        |
|---------------------|---------------------------------------------------------------------|
| `model.py`          | `FertilizerRAS` class and `run_ras` function (the 4-phase algorithm) |
| `preprocessing.py`  | FAOSTAT loaders, country filtering, shock application               |
| `postprocessing.py` | Summary tables, baseline-vs-shock comparison, CSV export            |
| `utils.py`          | Plotly + matplotlib helpers (Sankey, heatmap, comparison dashboard) |

For the equations, see `../docs/methodology.md`.
