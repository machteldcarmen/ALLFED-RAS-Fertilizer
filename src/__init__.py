"""Global fertilizer trade inertia model (RAS / IPF) for N, P, K.

The interface lives in :mod:`src.model`.  A typical minimal run looks like::

    from src.model import FertilizerRAS
    model = FertilizerRAS(P, C, T0)
    result = model.run()
    print(result.F)   # final fertilizer availability per country
    print(result.X)   # post-shock bilateral trade matrix
"""

from .model import FertilizerRAS, RASResult, run_ras
from .utils import use_allfed_style

__all__ = ["FertilizerRAS", "RASResult", "run_ras", "use_allfed_style"]
