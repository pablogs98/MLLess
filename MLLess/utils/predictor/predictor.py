# Authors: Marc Sanchez Artigas <marc.sanchez@urv.cat>
# License: BSD 3 clause
from pandas import Series
import numpy as np
from typing import Callable, Dict, Tuple
from collections.abc import Iterable
from scipy.optimize import curve_fit
from itertools import groupby
from MLLess.utils.predictor.outliers import rm_outlier

def rtol(x, p):
    r"""The relative error for current prediction:
      %rtol = (x - p)/|x|*100.
    """
    return 100*(x - p)/x

def stop_training(loss, context):
    threshold = context.get('threshold')
    if threshold:
        return loss < threshold
    else:
        loss_old = context.get('loss_old')
        if loss_old is None:
            loss_old = loss
        else:
            loss_delta = (loss_old - loss)/loss_old
            tol = context.get('tol')
            if loss_delta < tol:
                return True
            context['loss_old'] = loss
            return False

def ewa_stop_training(loss, context):
    # Alpha is a parameter allowed to change at each step to account
    # for batch_size changes, etc.
    ewa_loss, alpha = context.get('ewa_loss'), context.get('alpha')
    if ewa_loss is None:
        ewa_loss = loss
    else:
        ewa_loss = ewa_loss*(1.-alpha) + loss*alpha

    # Early stopping heuristic due to lack of improvement
    # on smoothed loss
    ewa_loss_min = context.get('ewa_loss_min')
    no_improvement = context.get('no_improvement', 0)
    if ewa_loss_min is None or ewa_loss < ewa_loss_min:
        no_improvement = 0
        ewa_loss_min = ewa_loss
    else:
        no_improvement += 1

    max_no_improvement = context.get('max_no_improvement')
    if (max_no_improvement is not None
           and no_improvement >= max_no_improvement):
         return True
    context['ewa_loss'] = ewa_loss
    context['ewa_loss_min'] = ewa_loss_min
    context['no_improvement'] = no_improvement
    return False

_fb = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

def _f(x: float,        # IN
        coef0: float,   # IN
        coef1: float,   # IN
        coef2: float,   # IN
        coef3: float,   # IN
        ) -> float:     # OUT
        return 1/(coef0*np.power(x, coef2) + coef1) + coef3

_gb = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

def _g(x: float,       # IN
       coef0: float,   # IN
       coef1: float,   # IN
       coef2: float,   # IN
       coef3: float,   # IN
       ) -> float:     # OUT
       return 1./(coef0*x**2 + coef1*x + coef2) + coef3

class LearningCurve(object):

    def __init__(self):
        self.x_, self.loss_ = [], []

    def clear(self):
        self.x_, self.loss_ = [], []

    def update(self, x, loss):
        self.x_.extend(x) if isinstance(x, Iterable) else self.x_.append(x)
        self.loss_.extend(loss) if isinstance(loss, Iterable) else self.loss_.append(loss)

    def step(self):
        return self.x_[-1]

    def fit(self, ewma=False):
        if ewma:
            df = Series(self.loss_)
            fwd = Series.ewm(df,span=5).mean() # take EWMA in fwd direction
            bwd = Series.ewm(df[::-1],span=5).mean() # take EWMA in bwd direction
            self.y_ = np.mean(np.vstack(( fwd, bwd[::-1] )), axis=0)
        else:
            self.y_ = self.loss_

        # Weight fit of the learning curve
        sigma = [1./_ for _ in np.arange(1, len(self.y_) + 1)]

        self.coef, _ = curve_fit(_f,
                            self.x_,
                            self.y_,
                            sigma=sigma,
                            absolute_sigma=True,
                            bounds=_fb,
                            maxfev=100000)

    def partial_fit(self):
        # Fit the learning curve
        self.coef, _ = curve_fit(_g,
                        self.x_,
                        self.loss_,
                        bounds=_gb,
                        maxfev=100000)

        self.perr = np.sqrt(np.diag(_))

    def estimate(self,
            max_steps: int,                     # IN
            context: Dict,                      # IN
            func: Callable=_f,                  # IN
            evaluate: Callable=stop_training,   # IN
            ) -> Tuple[int, float]:             # OUT
        n_steps = len(self.loss_)
        if n_steps >= max_steps:
            return n_steps, self.loss_[-1]
        if n_steps >= 3:
            step = 0
            while step < max_steps:
                loss = func(step, *self.coef)
                if evaluate(loss, context):
                    return step, loss
                step += 1
            return step, loss
        else:
            return -1, -1.

class SpeedCurve(object):

    def __init__(self,
                    ini_workers: int,     # IN
                ):
        self.ini_workers_ = ini_workers
        self.num_workers_ = ini_workers
        self.speed_ = { _ : [] for _ in range(1, ini_workers + 1) }
        self.dirty_ = { _ : False for _ in range(1, ini_workers + 1) }
        self.speed_wo_ = { _ : None for _ in range(1, ini_workers + 1) }

    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    def f(self,
            w: float,       # IN
            coef0: float,   # IN
            coef1: float,   # IN
            coef2: float,   # IN
            ) -> float:     # OUT
            return coef0 + (w-1)*coef1 + w*coef2

    def update(self, w, speed):
        self.speed_[w].append(speed)
        self.dirty_[w] = True
        self.num_workers_ = min(self.num_workers_, w)

    def extend(self, w, speed):
        if not isinstance(w, Iterable):
            raise TypeError("Expected an Iterable, found {}".format(w.__clas__))
        if not isinstance(speed, Iterable):
            raise TypeError("Expected an Iterable, found {}".format(speed.__class__))
        for _ in zip(w, speed):
            self.update(*_)

    def _filter_outliers(self, to_filter, threshold=0.1, window=10):
        for idx in to_filter:
            oloc = rm_outlier(self.speed_[idx],
                                    threshold=threshold,
                                    window=window)
            self.speed_wo_[idx] = np.array([self.speed_[idx][_] for _ in oloc.index if not oloc[_]])

    def fit(self):
        dirty_idx_ = [idx for idx, _ in self.dirty_.items() if self.dirty_[idx]]

        # Remove outliers
        self._filter_outliers(dirty_idx_, threshold=0.05)

        self.w_wo, self.speed_wo = np.array([], dtype='int'), np.array([], dtype='float')
        for idx in dirty_idx_:
            self.speed_wo = np.concatenate((self.speed_wo, self.speed_wo_[idx]))
            self.w_wo = np.concatenate((self.w_wo, np.full(self.speed_wo_[idx].shape, idx)))

        # Fit the learning curve
        self.coef, _ = curve_fit(self.f,
                                self.w_wo,
                                self.speed_wo,
                                bounds=self.bounds)

    def _extend(self, w, speed, window=10):
        count = [len(list(group)) for _, group in groupby(w)]
        del count[-1]
        count = np.cumsum(count)
        nloc = np.array([np.arange(_ , _ + window) for _ in count])
        nloc = nloc.flatten()
        w_, speed_, = np.delete(w, nloc), np.delete(speed, nloc)
        self.extend(w_, speed_)

    def estimate(self,
                    w: int,        # IN
                    ) -> float:     # OUT
        return 1./self.f(w, *self.coef)
