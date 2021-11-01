import numpy as np
from MLLess.algorithms.base import Base
from MLLess.models.cython_wrappers import get_mf_model


class ProbabilisticMatrixFactorisation(Base):
    def __init__(self, n_workers, n_users, n_items, n_factors=15, init_mean=.0,
                 init_std_dev=.1, lambda_=0.1, significance_threshold=0.05, momentum=0.8,
                 learning_rate_l=None, learning_rate_r=None, lambda_l=None, lambda_r=None, **kwargs):
        super().__init__(n_workers, **kwargs)

        self.n_factors = n_factors
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.learning_rate_l = learning_rate_l if learning_rate_l is not None else self.learning_rate
        self.learning_rate_r = learning_rate_r if learning_rate_r is not None else self.learning_rate
        self.lambda_l = lambda_l if lambda_l is not None else lambda_
        self.lambda_r = lambda_r if lambda_r is not None else lambda_
        self.momentum = momentum

        self.n_items = n_items
        self.n_users = n_users
        np.random.seed(self.seed)
        # self.L = np.random.normal(init_mean, init_std_dev, (n_users, n_factors))
        # self.R = np.random.normal(init_mean, init_std_dev, (n_items, n_factors))

        self.asp_threshold = significance_threshold

    def _initialise(self, trainset, n_minibatches):
        payload = []
        super_args, sup_super_args = super()._initialise(trainset, n_minibatches)

        args = {'n_factors': self.n_factors, 'init_mean': self.init_mean, 'init_std_dev': self.init_std_dev,
                'learning_rate_l': self.learning_rate_l, 'learning_rate_r': self.learning_rate_r,
                'momentum': self.momentum, 'lambda_l': self.lambda_l, 'lambda_r': self.lambda_r,
                'n_users': self.n_users, 'n_items': self.n_items, 'model': get_mf_model}
        args = {**args, **super_args}

        for p in range(self.n_workers):
            args['worker_id'] = p
            payload.append({'args': args.copy()})

        return payload, sup_super_args
