import numpy as np

DTYPE = np.float32


class MatrixFactorisation:
    def __init__(self, **kwargs):
        self.L_size = kwargs['n_users']
        self.R_size = kwargs['n_items']
        self.L_learning_rate = kwargs['learning_rate_l']
        self.R_learning_rate = kwargs['learning_rate_r']
        self.n_factors = kwargs['n_factors']
        self.lambda_l = kwargs['lambda_l']
        self.lambda_r = kwargs['lambda_r']
        self.momentum = kwargs['momentum']
        self.worker_id = kwargs['worker_id']

        self.loss = 0.0
        np.random.seed(kwargs['seed'])

        # model
        self.L = np.random.normal(.0, .1, (self.L_size, self.n_factors)).astype(DTYPE)
        self.R = np.random.normal(.0, .1, (self.R_size, self.n_factors)).astype(DTYPE)

        # Biases
        self.L_bias = np.random.normal(.0, .1, self.L_size).astype(DTYPE)
        self.R_bias = np.random.normal(.0, .1, self.R_size).astype(DTYPE)

        # Momentum
        self.L_velocity = np.zeros((self.L_size, self.n_factors), dtype=DTYPE)
        self.R_velocity = np.zeros((self.R_size, self.n_factors), dtype=DTYPE)
        self.L_bias_velocity = np.zeros(self.L_size, dtype=DTYPE)
        self.R_bias_velocity = np.zeros(self.R_size, dtype=DTYPE)

    def step(self, epoch, _, minibatch):
        self._sgd(epoch, minibatch)

    def _sgd(self, _, minibatch):
        minibatch_size = minibatch.shape[0]
        L_update = np.zeros((self.L_size, self.n_factors), dtype=DTYPE)
        R_update = np.zeros((self.R_size, self.n_factors), dtype=DTYPE)

        L_bias_update = np.zeros(self.L_size, dtype=DTYPE)
        R_bias_update = np.zeros(self.R_size, dtype=DTYPE)

        ratings = minibatch[:, 2].astype(DTYPE)

        L_learning_rate = self.L_learning_rate
        R_learning_rate = self.R_learning_rate

        self.minibatch_rows = minibatch_rows = minibatch[:, 0].astype(int)
        self.minibatch_cols = minibatch_cols = minibatch[:, 1].astype(int)

        # compute objective function
        pred = np.sum(np.multiply(self.L[minibatch_rows, :], self.R[minibatch_cols, :]), axis=1)
        diff = pred + self.L_bias[minibatch_rows] + self.R_bias[minibatch_cols] - ratings

        # compute gradients & nesterov momentum
        L_gradient = 2.0 * np.multiply(diff[:, np.newaxis], self.R[minibatch_cols, :])
        R_gradient = 2.0 * np.multiply(diff[:, np.newaxis], self.L[minibatch_rows, :])
        self.L_velocity[minibatch_rows, :] = (self.momentum * self.L_velocity[minibatch_rows, :]) + L_gradient / minibatch_size
        self.R_velocity[minibatch_cols, :] = (self.momentum * self.R_velocity[minibatch_cols, :]) + R_gradient / minibatch_size
        L_nesterov_update = (self.momentum * self.L_velocity[minibatch_rows, :]) + L_gradient / minibatch_size
        R_nesterov_update = (self.momentum * self.R_velocity[minibatch_cols, :]) + R_gradient / minibatch_size

        L_bias_gradient = R_bias_gradient = 2.0 * diff
        self.L_bias_velocity[minibatch_rows] = (self.momentum * self.L_bias_velocity[minibatch_rows]) + L_bias_gradient / minibatch_size
        self.R_bias_velocity[minibatch_cols] = (self.momentum * self.R_bias_velocity[minibatch_cols]) + R_bias_gradient / minibatch_size
        L_bias_nesterov_update = (self.momentum * self.L_bias_velocity[minibatch_rows]) + L_bias_gradient / minibatch_size
        R_bias_nesterov_update = (self.momentum * self.R_bias_velocity[minibatch_cols]) + R_bias_gradient / minibatch_size

        # accumulate updates
        for i in range(minibatch_size):
            L_update[minibatch_rows[i], :] += L_nesterov_update[i, :]
            R_update[minibatch_cols[i], :] += R_nesterov_update[i, :]
            L_bias_update[minibatch_rows[i]] += L_bias_nesterov_update[i]
            R_bias_update[minibatch_cols[i]] += R_bias_nesterov_update[i]

        # update model
        self.L = self.L - L_learning_rate * L_update
        self.L_bias = self.L_bias - L_learning_rate * L_bias_update
        self.R = self.R - R_learning_rate * R_update
        self.R_bias = self.R_bias - R_learning_rate * R_bias_update

        # compute loss
        loss = np.linalg.norm(diff) ** 2
        self.loss = (loss, minibatch_size)

    def loss_function(self, rows, cols, ratings, pred):
        return np.sum(pred + self.L_bias[rows] + self.R_bias[cols] - ratings)