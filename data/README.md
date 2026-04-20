# Data

The FAOSTAT bulk extracts used by the model are **large** (hundreds of
MB) and therefore are **not** committed to this repository — see the
`.gitignore` at the project root.

Download them from FAOSTAT and place them under this folder with the
following layout. The loaders in `src/preprocessing.py` expect exactly
these names.

## Required downloads

| File                                                          | FAOSTAT domain                         | Place under                                        |
|---------------------------------------------------------------|----------------------------------------|----------------------------------------------------|
| `Inputs_FertilizersNutrient_E_All_Data_NOFLAG.csv`            | Inputs → Fertilizers by Nutrient       | `data/Inputs_FertilizersNutrient_E_All_Data/`      |
| `Fertilizers_DetailedTradeMatrix_E_All_Data_NOFLAG.csv`       | Trade → Detailed trade matrix          | `data/Fertilizers_DetailedTradeMatrix_E_All_Data/` |

Go to https://www.fao.org/faostat/, search for the domain name, and use
the "**Bulk download** → *All Data (Normalized)*" option. The `_NOFLAG`
variant (no flag columns) is the smallest and what the loaders expect.

## What the loaders read

`src/preprocessing.py` uses these FAOSTAT codes:

| Item            | Item Code |
|-----------------|-----------|
| Nitrogen (N)    | 3102      |
| Phosphate (P₂O₅)| 3103      |
| Potash (K₂O)    | 3104      |

| Element     | Element Code |
|-------------|--------------|
| Production  | 5510         |
| Import      | 5610         |
| Export      | 5910         |
| AgUse       | 5157         |

## Scenario files

Put custom shock / scenario CSVs under `data/scenario_files/`. Each
scenario is just a mapping from country name to surviving production
fraction; see the example in `scripts/real_data_nitrogen.ipynb`.
