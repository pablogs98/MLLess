# Authors: Marc Sanchez Artigas <marc.sanchez@urv.cat>
# License: BSD 3 clause
import numpy as np
import pandas as pd
from collections.abc import Iterable

class KneeLocator( object ):
    r"""
        Provide first derivative knee locator for
        convex decreasing functions.

        Parameters
        ----------
        threshold: float, optional
            Specify the tolerance to the tangency.
        max_no_improvement: int, optinal
            Specifiy the consecutive number of steps to
            consider a candidate tangency as 'knee'.
    """
    def __init__(self,
                threshold: float = -0.0001,          # IN
                max_no_improvement: int = 5,         # IN
            ):
            self.threshold = threshold
            if self.threshold is not None and self.threshold >= 0:
                raise ValueError(
                    f"treshold should be < 0, got "
                        f"{self.threshold} instead.")

            self.max_no_improvement = max_no_improvement
            if self.max_no_improvement is not None and self.max_no_improvement < 0:
                raise ValueError(
                    f"max_no_improvement should be >= 0, got "
                        f"{self.max_no_improvement} instead.")

    def ewma_(self, x):
        df = pd.Series(x)
        fwd = pd.Series.ewm(df,span=20).mean()         # take EWMA in fwd direction
        bwd = pd.Series.ewm(df[::-1],span=20).mean()   # take EWMA in bwd direction
        return np.mean(np.vstack(( fwd, bwd[::-1] )), axis=0)

    def find(self,
                loss: Iterable        # IN
            ) -> int:                 # OUT

        n_values = len(loss)
        if n_values >= 3:
            diff = np.diff(self.ewma_(loss))   # compute first derivative
            diff = self.ewma_(diff)
            min_curvature_idx = np.argmax(diff > self.threshold)
            if min_curvature_idx:
                if not hasattr(self, 'min_curvature_idx') or \
                        (min_curvature_idx > self.min_curvature_idx):
                    self.min_curvature_idx = min_curvature_idx
                    self.no_improvement = 0
                    return 0
                self.no_improvement += 1
                if self.no_improvement >= self.max_no_improvement:
                    return min_curvature_idx
            else:
                if hasattr(self, 'min_curvature_idx'):
                    del self.min_curvature_idx
        return 0
