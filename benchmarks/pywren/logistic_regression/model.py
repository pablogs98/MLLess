from cmath import sqrt, exp
from math import log1p, log

import numpy as np

DTYPE = np.float64


class LogisticRegression:
    """
    Logistic Regression Model class.
    """

    def __init__(self, **kwargs):
        # logistic regression params
        self.num_features = kwargs['num_features']
        self.biased = kwargs['biased']
        self.learning_rate = kwargs['learning_rate']
        self.reg_param = kwargs['reg_param']
        self.worker_id = kwargs['worker_id']
        np.random.seed(kwargs['seed'])

        # Adam
        self.beta_1 = kwargs['beta_1']  # momentum factor 1
        self.beta_2 = kwargs['beta_2']  # momentum factor 2
        self.m_momentum = np.zeros(self.num_features, DTYPE)
        self.v_momentum = np.zeros(self.num_features, DTYPE)
        self.bias_m_momentum = 0.0
        self.bias_v_momentum = 0.0

        # Model
        k = 1.0 / self.num_features
        self.weights = np.random.uniform(-k, k, self.num_features).astype(DTYPE)
        self.bias_term = np.random.uniform(-k, k, 1).astype(DTYPE)[0]

        # Gaia accumulated gradient
        self.acc_gradient = np.zeros(self.num_features, DTYPE)

    def step(self, epoch, step, minibatch):
        self._adam(epoch, step, minibatch)

    def _adam(self, epoch, step, minibatch):
        minibatch_size = minibatch.shape[0]
        learning_rate = self.learning_rate / sqrt(epoch + 1)

        gradient = np.zeros(self.num_features, DTYPE)
        pred = np.zeros(minibatch_size, DTYPE)

        minibatch_samples = minibatch[:, 0:self.num_features].astype(np.float64)
        minibatch_labels = minibatch[:, self.num_features].astype(np.float32)

        bias_gradient = 0.0
        loss = 0.0

        for m in range(minibatch_size):
            pred[m] = self.prediction(minibatch_samples[m])
            loss += loss_function(pred[m], minibatch_labels[m])

        # Calculate gradients
        for m in range(minibatch_size):
            for f in range(self.num_features):
                gradient[f] += (pred[m] - minibatch_labels[m]) * minibatch_samples[m, f]

            if self.biased:
                bias_gradient += (pred[m] - minibatch_labels[m])

        # update weights with accumulated gradient
        self.model_update(gradient, bias_gradient, learning_rate, step, minibatch_size)

        self.loss = (loss, minibatch_size)

    def model_update(self, gradient, bias_gradient, learning_rate, step, minibatch_size):
        for f in range(self.num_features):
            reg_grad = (gradient[f] / minibatch_size) * self.reg_param
            self.m_momentum[f] = self.momentum_update(self.m_momentum[f], reg_grad)
            self.v_momentum[f] = self.sqr_momentum_update(self.v_momentum[f], reg_grad)
            update = self.update_param(self.m_momentum[f], self.v_momentum[f], learning_rate, step)
            self.weights[f] -= update
            self.acc_gradient[f] -= update

        # update bias term
        if self.biased:
            bias_reg_grad = (bias_gradient / minibatch_size) * self.reg_param
            self.bias_m_momentum = self.momentum_update(self.bias_m_momentum, bias_reg_grad)
            self.bias_v_momentum = self.sqr_momentum_update(self.bias_v_momentum, bias_reg_grad)
            self.bias_term -= self.update_param(self.bias_m_momentum, self.bias_v_momentum, learning_rate, step)

    def momentum_update(self, momentum, grad):
        return self.beta_1 * momentum + (1 - self.beta_1) * grad

    def sqr_momentum_update(self, momentum, grad):
        return self.beta_2 * momentum + (1 - self.beta_2) * pow(grad, 2)

    def update_param(self, m_momentum, v_momentum, learning_rate, step):
        bias_correction1 = 1 - pow(self.beta_1, step + 1)
        bias_correction2 = 1 - pow(self.beta_2, step + 1)
        step_size = learning_rate / bias_correction1
        epsilon = pow(10, -9)
        denom = sqrt(v_momentum) / (sqrt(bias_correction2) + epsilon)
        return step_size * (m_momentum / denom)

    def prediction(self, sample):
        # Kahan sumation
        pred = np.float64(0.)
        c = np.float64(0.)

        for i in range(self.num_features):
            feature_pred = self.weights[i] * sample[i]
            y = feature_pred - c
            t = pred + y
            c = (t - pred) - y
            pred = t

        if self.biased:
            pred += self.bias_term
        pred = sigmoid(pred)
        return pred.real


def loss_function(pred, label):
    if label == 0.0:
        loss = -log1p(-pred + 10 ** -9)
    else:
        loss = -log(pred + 10 ** -9)
    return loss


def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))  # sigmoid
