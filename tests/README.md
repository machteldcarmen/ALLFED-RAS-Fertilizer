# tests

Unit tests for the package. Run locally with:

```bash
pytest
```

These tests also run on every push via the GitHub Action in
`../.github/workflows/testing.yml`.

| File            | What it covers                                                    |
|-----------------|-------------------------------------------------------------------|
| `test_ras.py`   | RAS target matching, zero-structure preservation, Phase 1 mass balance, convergence, degenerate inputs |
