# Authors: Marc Sanchez Artigas <marc.sanchez@urv.cat>
# License: BSD 3 clause
import abc
from MLLess.utils.predictor.predictor import _f, _g, LearningCurve, SpeedCurve, rtol
from MLLess.utils.predictor.knee import KneeLocator
from typing import Dict

class Action:
    pass

class ReclaimAction(Action):
    def __init__(self, n_workers:int=1):
        self.n_workers = n_workers

class Scheduler(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '_schedule') and
            callable(subclass._schedule) or
                NotImplemented)

    @abc.abstractmethod
    def schedule(self,
                context: Dict,           # IN
                verbose: bool = False    # IN
            ) -> Action:
        raise NotImplementedError

class WorkerScheduler(Scheduler):
    r"""
        Scheduler to scale down the number of workers.

        Parameters
        ----------
        ini_workers: int,
            Specify initial number of workers.
        noise_window_length: int, optional
            Set the number noisy steps to ignore.
        curvature_tol: float, optional
            Specify the tolerance to the tangency.
        max_no_improvement: int, optinal
            Specifiy the consecutive number of steps to
            consider a candidate tangency as 'knee'.
        delta: float, optional
            Control the projection timespan in seconds.
        tol: float, optional
            Set the tolerance to the maximum convergence deviation.
    """
    def __init__(self,
                ini_workers: int,                    # IN
                noise_window_length: int = 5,        # IN
                curvature_tol: float = -0.000075,      # IN
                max_no_improvement: int = 5,         # IN
                delta: float = 10,                   # IN
                tol: float = 1                       # IN
                ):

        self.n_workers_ = ini_workers
        self.noise_window_length_ = noise_window_length
        self.kl = KneeLocator(curvature_tol, max_no_improvement)
        self.delta = delta
        self.tol = tol

        self.lc = LearningCurve()
        self.sc = SpeedCurve(self.n_workers_)
        self.skip = 0
        self.ini_workers = ini_workers

    def update(self,
            step: int,                  # IN
            loss: float,                # IN
            step_duration: float):      # IN

        self.lc.update(step, loss)
        if self.skip:
            self.skip -= 1
        else:
            self.sc.update(self.n_workers_, step_duration)

    def is_in_stationary_regime(self, verbose=False):
        is_stationary = hasattr(self, 'stationary_')
        if not is_stationary:
            knee = self.kl.find(self.lc.loss_)
            if knee:
                self.stationary_ = knee
                if verbose:
                    print(f"Stationary regime at step {self.stationary_}")
        return hasattr(self, 'stationary_')

    def _fit_base_model(self):
        self.lc.fit()
        self.sc.fit()
        self.coeff_base_ = self.lc.coef
        self.speed_base_ = self.sc.estimate(self.n_workers_)

    def _predict_loss_decay(self, current_step):
        # compute prediction for the initial number of workers
        forward_steps = int(self.speed_base_*self.delta + current_step)
        pred_loss_base = _f(forward_steps, *self.coeff_base_)

        # compute prediction for the current number of workers
        self.lc.partial_fit()
        self.sc.fit()
        speed = self.sc.estimate(self.n_workers_)
        forward_steps = int(speed*self.delta + current_step)
        pred_loss = _g(forward_steps, *self.lc.coef)

        return pred_loss_base, pred_loss

    def schedule(self,
                context: Dict,           # IN
                verbose: bool = False    # IN
            ) -> Action:

        if self.n_workers_ < 2:
             return None

        is_stationary = hasattr(self, 'stationary_')
        if not is_stationary:
            return None

        is_first_call = not hasattr(self, 'coeff_base_')
        if is_first_call:
            self._fit_base_model()
            return ReclaimAction()
        else:
            current_step = context.get('current_step')
            pred_loss_base, pred_loss = self._predict_loss_decay(current_step)
            re = rtol(pred_loss_base, pred_loss)
            if verbose:
                print('Predicted loss {:4.4f} after {:2d} s, rtol {:4.4f}%' \
                        .format(pred_loss, self.delta, re))
            if re > -self.tol:
                return ReclaimAction()
        return None

    def scale_down(self, n_workers:int=1, verbose=False):
        if not n_workers or n_workers < 0:
            return
        self.lc = LearningCurve()
        self.n_workers_ = max(1, self.n_workers_ - n_workers)
        self.skip = self.noise_window_length_ - 1
        if verbose:
            print(f"Number of workers {self.n_workers_}")
