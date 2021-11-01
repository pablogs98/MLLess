cimport cython
cimport numpy as cnp
from libc.math cimport sqrt, exp, log1p, pow, log

import numpy as np
from lithops.libs.model cimport CModel
from lithops.libs.model import Model

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

class LogisticRegressionModel(Model):
    def __init__(self, **kwargs):
        super().__init__(CLogisticRegressionModel.__new__(CLogisticRegressionModel, **kwargs))

cdef class CLogisticRegressionModel(CModel):
    """
    Logistic Regression Model class.
    """

    cdef int num_features
    cdef bint biased

    cdef DTYPE_t learning_rate
    cdef DTYPE_t reg_param
    cdef DTYPE_t bias_term

    cdef DTYPE_t beta_1
    cdef DTYPE_t beta_2
    cdef DTYPE_t[::1] m_momentum
    cdef DTYPE_t[::1] v_momentum
    cdef DTYPE_t bias_m_momentum
    cdef DTYPE_t bias_v_momentum

    cdef DTYPE_t[::1] _weights
    cdef DTYPE_t[::1] acc_gradient

    def __cinit__(self, **kwargs):
        # logistic regression params
        self.num_features = kwargs['num_features']
        self.biased = kwargs['biased']
        self.learning_rate = kwargs['learning_rate']
        self.reg_param = kwargs['reg_param']
        np.random.seed(kwargs['seed'])

        # Adam
        self.beta_1 = kwargs['beta_1']      # momentum factor 1
        self.beta_2 = kwargs['beta_2']      # momentum factor 2
        self.m_momentum = np.zeros(self.num_features, DTYPE)
        self.v_momentum = np.zeros(self.num_features, DTYPE)
        self.bias_m_momentum = 0.0
        self.bias_v_momentum = 0.0

        # Model
        cdef DTYPE_t k = 1.0 / self.num_features
        self._weights = np.random.uniform(-k, k, self.num_features).astype(DTYPE)
        self.bias_term = np.random.uniform(-k, k, 1)[0]

        # Gaia accumulated gradient
        self.acc_gradient = np.zeros(self.num_features, DTYPE)


    @cython.initializedcheck(False)
    cpdef tuple step(self, int epoch, int step, object minibatch):
        return self._adam(epoch, step, minibatch)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef tuple _adam(self, int epoch, int step, object minibatch):
        cdef Py_ssize_t f, m

        cdef Py_ssize_t minibatch_size = minibatch.shape[0]
        cdef DTYPE_t learning_rate = self.learning_rate / sqrt(epoch + 1)
        cdef DTYPE_t threshold = self.threshold / sqrt(step + 1)

        cdef DTYPE_t[::1] gradient = np.zeros(self.num_features, DTYPE)
        cdef DTYPE_t[::1] pred = np.zeros(minibatch_size, DTYPE)

        cdef cnp.float_t[:, ::1] minibatch_samples = minibatch[:, 0:self.num_features].astype(np.float64)
        cdef cnp.float32_t[::1] minibatch_labels = minibatch[:, self.num_features].astype(np.float32)

        cdef DTYPE_t bias_gradient = 0.0
        cdef DTYPE_t loss = 0.0

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

        return loss, minibatch_size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void model_update(self, DTYPE_t[::1] gradient, DTYPE_t bias_gradient, DTYPE_t learning_rate, int step, int minibatch_size):
        cdef Py_ssize_t f
        cdef DTYPE_t reg_grad, bias_reg_grad, update

        for f in range(self.num_features):
            reg_grad = (gradient[f] / minibatch_size) * self.reg_param
            self.m_momentum[f] = self.momentum_update(self.m_momentum[f], reg_grad)
            self.v_momentum[f] = self.sqr_momentum_update(self.v_momentum[f], reg_grad)
            update = self.update_param(self.m_momentum[f], self.v_momentum[f], learning_rate, step)
            self._weights[f] -= update
            self.acc_gradient[f] -= update

        # update bias term
        if self.biased:
            bias_reg_grad = (bias_gradient / minibatch_size) * self.reg_param
            self.bias_m_momentum = self.momentum_update(self.bias_m_momentum, bias_reg_grad)
            self.bias_v_momentum = self.sqr_momentum_update(self.bias_v_momentum, bias_reg_grad)
            self.bias_term -= self.update_param(self.bias_m_momentum, self.bias_v_momentum, learning_rate, step)

    @cython.initializedcheck(False)
    cdef DTYPE_t momentum_update(self, DTYPE_t momentum, DTYPE_t grad):
        return self.beta_1 * momentum + (1 - self.beta_1) * grad

    @cython.initializedcheck(False)
    cdef DTYPE_t sqr_momentum_update(self, DTYPE_t momentum, DTYPE_t grad):
        return self.beta_2 * momentum + (1 - self.beta_2) * pow(grad, 2)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef DTYPE_t update_param(self, DTYPE_t m_momentum, DTYPE_t v_momentum, DTYPE_t learning_rate, int step):
        cdef DTYPE_t bias_correction1 = 1 - pow(self.beta_1, step + 1)
        cdef DTYPE_t bias_correction2 = 1 - pow(self.beta_2, step + 1)
        cdef DTYPE_t step_size = learning_rate / bias_correction1
        cdef cnp.float_t epsilon = pow(10, -9)
        cdef DTYPE_t denom = sqrt(v_momentum) / (sqrt(bias_correction2) + epsilon)
        return step_size * (m_momentum / (denom + epsilon))

    @cython.initializedcheck(False)
    cpdef tuple get_significant_updates(self, int step):
        return self._get_significant_updates(step)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef tuple _get_significant_updates(self, int step):
        cdef Py_ssize_t significant = 0
        cdef tuple update
        cdef Py_ssize_t[::1] u_col = np.empty(self.num_features, dtype=int)
        cdef DTYPE_t[::1] u_data = np.empty(self.num_features, dtype=DTYPE)
        cdef DTYPE_t threshold = self.threshold / sqrt(step + 1)

        for f in range(self.num_features):
            if abs(self.acc_gradient[f] / self._weights[f]) > threshold:
                u_col[significant] = f
                u_data[significant] = self.acc_gradient[f]
                significant += 1
                self.acc_gradient[f] = 0.0
        self.significance = significant

        # prepare significant updates to be sent
        if significant != 0:
            update = (np.array(u_data[:significant]), np.array(u_col[:significant]))
            return update
        else:
            return None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef void aggregate_updates(self, tuple content):
        self._aggregate_updates(content[0], content[1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _aggregate_updates(self, DTYPE_t[::1] data, Py_ssize_t[::1] cols):
        cdef Py_ssize_t update_size = len(data)
        cdef Py_ssize_t l

        for l in range(update_size):
            self._weights[cols[l]] += data[l]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef void aggregate_model(self, list content):
        self._aggregate_model(content[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void _aggregate_model(self, DTYPE_t[::1] model):
        cdef Py_ssize_t i

        for i in range(self.num_features):
            self._weights[i] = (self._weights[i] + model[i]) / 2.0

    @property
    def weights(self):
        return [np.array(self._weights)]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef DTYPE_t prediction(self, cnp.float_t[::1] sample):
        cdef Py_ssize_t i

        # Kahan sumation
        cdef cnp.float_t pred = np.float64(0.)
        cdef cnp.float_t feature_pred
        cdef cnp.float_t c = np.float64(0.)
        cdef cnp.float_t t, y

        for i in range(self.num_features):
            feature_pred = self._weights[i] * sample[i]
            y = feature_pred - c
            t = pred + y
            c = (t - pred) - y
            pred = t
        if self.biased:
            pred += self.bias_term
        pred = sigmoid(pred)

        return pred


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef DTYPE_t loss_function(DTYPE_t pred, cnp.float32_t label):
    cdef DTYPE_t loss
    if label == 0.0:
        loss = -log1p(-pred + pow(10, -9))
    else:
        loss = -log(pred + pow(10, -9))
    return loss

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef DTYPE_t sigmoid(DTYPE_t z):
    return 1.0 / (1.0 + exp(-z))  # sigmoid
