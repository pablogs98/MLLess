from MLLess.algorithms.base import Base
from MLLess.models.cython_wrappers import get_lr_model, get_sparse_lr_model


class LogisticRegression(Base):
    """
    Logistic Regression SGD
    """

    def __init__(self, n_workers, num_features, biased=True, reg_param=0.01,
                 beta_1=0.9, beta_2=0.999, **kwargs):
        super().__init__(n_workers, **kwargs)

        self.biased = biased
        self.reg_param = reg_param
        self.num_features = num_features
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        if self.biased:
            self.gradient_size = self.num_features + 1
        else:
            self.gradient_size = self.num_features

    def _initialise(self, trainset, n_minibatches):
        payload = []
        super_args, sup_super_args = super()._initialise(trainset, n_minibatches)

        args = {'num_features': self.num_features, 'learning_rate': self.learning_rate, 'reg_param': self.reg_param,
                'biased': self.biased, 'beta_1': self.beta_1, 'beta_2': self.beta_2, 'model': get_lr_model}
        args = {**args, **super_args}

        for p in range(self.n_workers):
            args['worker_id'] = p
            payload.append({'args': args.copy()})

        return payload, sup_super_args


class SparseLogisticRegression(LogisticRegression):
    def __init__(self, n_workers, num_features, **kwargs):
        super().__init__(n_workers, num_features, **kwargs)

    def _initialise(self, trainset, n_minibatches):
        payload, sup_super_args = super()._initialise(trainset, n_minibatches)
        for p in payload:
            p['args']['model'] = get_sparse_lr_model
        return payload, sup_super_args
