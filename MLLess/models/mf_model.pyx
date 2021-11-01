cimport cython
cimport numpy
import numpy as np

from lithops.libs.model cimport CModel, DTYPE_t
from lithops.libs.model import Model

cdef extern from "math.h":
    double sqrt(double m)

DTYPE = np.float32

class MatrixFactorisationModel(Model):
    def __init__(self, **kwargs):
        super().__init__(CMatrixFactorisationModel.__new__(CMatrixFactorisationModel, **kwargs))


cdef class CMatrixFactorisationModel(CModel):
    cdef Py_ssize_t L_size
    cdef Py_ssize_t R_size
    cdef Py_ssize_t n_factors

    cdef DTYPE_t L_learning_rate
    cdef DTYPE_t R_learning_rate
    cdef DTYPE_t lambda_l
    cdef DTYPE_t lambda_r
    cdef DTYPE_t momentum

    # PMF model
    cdef DTYPE_t [:, ::1] L
    cdef DTYPE_t [:, ::1] R

    # Biases
    cdef DTYPE_t [::1] L_bias
    cdef DTYPE_t [::1] R_bias

    # Momentum
    cdef DTYPE_t [:, ::1] L_velocity
    cdef DTYPE_t [:, ::1] R_velocity

    cdef DTYPE_t [::1] L_bias_velocity
    cdef DTYPE_t [::1] R_bias_velocity

    # Gaia accumulative gradient
    cdef DTYPE_t [:, ::1] L_acc_gradient
    cdef DTYPE_t [:, ::1] R_acc_gradient
    cdef Py_ssize_t [::1] minibatch_rows
    cdef Py_ssize_t [::1] minibatch_cols

    @cython.initializedcheck(False)
    def __cinit__(self, **kwargs):
        self.L_size = kwargs['n_users']
        self.R_size = kwargs['n_items']
        self.L_learning_rate = kwargs['learning_rate_l']
        self.R_learning_rate = kwargs['learning_rate_r']
        self.n_factors = kwargs['n_factors']
        self.lambda_l = kwargs['lambda_l']
        self.lambda_r = kwargs['lambda_r']
        self.momentum = kwargs['momentum']
        init_mean = kwargs['init_mean']
        init_std_dev = kwargs['init_std_dev']
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

        # gaia cumulative updates
        self.L_acc_gradient = np.zeros((self.L_size, self.n_factors), dtype=DTYPE)
        self.R_acc_gradient = np.zeros((self.R_size, self.n_factors), dtype=DTYPE)

    @cython.initializedcheck(False)
    cpdef tuple step(self, int epoch, int _, object minibatch):
        return self._sgd(epoch, minibatch)

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef tuple _sgd(self, int epoch, object minibatch):
        cdef Py_ssize_t i, j, k, l, m
        cdef DTYPE_t loss

        cdef int minibatch_size = minibatch.shape[0]
        cdef DTYPE_t [:, ::1] L_update = np.zeros((self.L_size, self.n_factors), dtype=DTYPE)
        cdef DTYPE_t [:, ::1] R_update = np.zeros((self.R_size, self.n_factors), dtype=DTYPE)

        cdef DTYPE_t [::1] L_bias_update = np.zeros(self.L_size, dtype=DTYPE)
        cdef DTYPE_t [::1] R_bias_update = np.zeros(self.R_size, dtype=DTYPE)

        cdef DTYPE_t [:, ::1] L_gradient = np.zeros((minibatch_size, self.n_factors), dtype=DTYPE)
        cdef DTYPE_t [:, ::1] R_gradient = np.zeros((minibatch_size, self.n_factors), dtype=DTYPE)

        cdef DTYPE_t [::1] L_bias_gradient = np.zeros(minibatch_size, dtype=DTYPE)
        cdef DTYPE_t [::1] R_bias_gradient = np.zeros(minibatch_size, dtype=DTYPE)

        cdef DTYPE_t [:, ::1] L_nesterov_update = np.zeros((minibatch_size, self.n_factors), dtype=DTYPE)
        cdef DTYPE_t [:, ::1] R_nesterov_update = np.zeros((minibatch_size, self.n_factors), dtype=DTYPE)

        cdef DTYPE_t [::1] L_bias_nesterov_update = np.zeros(minibatch_size, dtype=DTYPE)
        cdef DTYPE_t [::1] R_bias_nesterov_update = np.zeros(minibatch_size, dtype=DTYPE)

        cdef DTYPE_t [::1] ratings = minibatch[:, 2].astype(DTYPE)

        cdef DTYPE_t [:, ::1] pred = np.zeros((minibatch_size, self.n_factors), dtype=DTYPE)
        cdef DTYPE_t [::1] diff = np.zeros(minibatch_size, dtype=DTYPE)

        cdef DTYPE_t L_learning_rate = self.L_learning_rate
        cdef DTYPE_t R_learning_rate = self.R_learning_rate

        cdef Py_ssize_t [::1] minibatch_rows
        cdef Py_ssize_t [::1] minibatch_cols

        self.minibatch_rows = minibatch_rows = minibatch[:, 0].astype(int)
        self.minibatch_cols = minibatch_cols = minibatch[:, 1].astype(int)

        # compute objective function
        pred = self.prediction(minibatch_rows, minibatch_cols, minibatch_size)
        for m in range(minibatch_size):
            i = minibatch_rows[m]
            j = minibatch_cols[m]
            for k in range(self.n_factors):
                diff[m] = diff[m] + pred[m][k]
            diff[m] += self.L_bias[i] + self.R_bias[j] - ratings[m]

        # compute loss
        loss = self.loss_function(minibatch_rows, minibatch_cols, ratings, pred, minibatch_size)

        # compute gradients & nesterov momentum
        for m in range(minibatch_size):
            i = minibatch_rows[m]
            j = minibatch_cols[m]
            for k in range(self.n_factors):
                L_gradient[m, k] = gradient(diff[m], self.R[j, k])
                R_gradient[m, k] = gradient(diff[m], self.L[i, k])
                self.L_velocity[i, k] = velocity(self.momentum, self.L_velocity[i, k], L_gradient[m, k], minibatch_size)
                self.R_velocity[j, k] = velocity(self.momentum, self.R_velocity[j, k], R_gradient[m, k], minibatch_size)
                L_nesterov_update[m, k] = velocity(self.momentum, self.L_velocity[i, k], L_gradient[m, k], minibatch_size)
                R_nesterov_update[m, k] = velocity(self.momentum, self.R_velocity[j, k], R_gradient[m, k], minibatch_size)
            L_bias_gradient[m] = 2 * diff[m]
            R_bias_gradient[m] = 2 * diff[m]
            self.L_bias_velocity[i] = velocity(self.momentum, self.L_bias_velocity[i], L_bias_gradient[m], minibatch_size)
            self.R_bias_velocity[j] = velocity(self.momentum, self.R_bias_velocity[j], R_bias_gradient[m], minibatch_size)
            L_bias_nesterov_update[m] = velocity(self.momentum, self.L_bias_velocity[i], L_bias_gradient[m], minibatch_size)
            R_bias_nesterov_update[m] = velocity(self.momentum, self.R_bias_velocity[j], R_bias_gradient[m], minibatch_size)

        # accumulate updates
        for m in range(minibatch_size):
            i = minibatch_rows[m]
            j = minibatch_cols[m]
            for k in range(self.n_factors):
                L_update[i, k] += L_nesterov_update[m, k]
                R_update[j, k] += R_nesterov_update[m, k]
            L_bias_update[i] += L_bias_nesterov_update[m]
            R_bias_update[j] += R_bias_nesterov_update[m]

        # update model
        for i in range(self.L_size):
            for k in range(self.n_factors):
                self.L[i, k] = model_update(self.L[i, k], L_learning_rate, L_update[i, k])
                self.L_acc_gradient[i, k] = model_update(self.L_acc_gradient[i, k], L_learning_rate, L_update[i, k])
            self.L_bias[i] = model_update(self.L_bias[i], L_learning_rate, L_bias_update[i])

        for j in range(self.R_size):
            for k in range(self.n_factors):
                self.R[j, k] = model_update(self.R[j, k], R_learning_rate, R_update[j, k])
                self.R_acc_gradient[j, k] = model_update(self.R_acc_gradient[j, k], R_learning_rate, R_update[j, k])
            self.R_bias[j] = model_update(self.R_bias[j], R_learning_rate, R_bias_update[j])

        return loss, minibatch_size

    @cython.initializedcheck(False)
    cpdef tuple get_significant_updates(self, Py_ssize_t step):
        return self._get_significant_updates(step)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef tuple _get_significant_updates(self, Py_ssize_t step):
        cdef Py_ssize_t[::1] l_u_col = np.empty(self.n_factors * self.L_size, dtype=int)
        cdef Py_ssize_t[::1] l_u_row = np.empty(self.n_factors * self.L_size, dtype=int)
        cdef DTYPE_t[::1] l_u_data = np.empty(self.n_factors * self.L_size, dtype=DTYPE)

        cdef Py_ssize_t[::1] r_u_col = np.empty(self.n_factors * self.L_size, dtype=int)
        cdef Py_ssize_t[::1] r_u_row = np.empty(self.n_factors * self.L_size, dtype=int)
        cdef DTYPE_t [::1] r_u_data = np.empty(self.n_factors * self.L_size, dtype=DTYPE)

        cdef dict unique_rows = {}
        cdef dict unique_cols = {}
        cdef DTYPE_t threshold = self.threshold / sqrt(step + 1)
        cdef Py_ssize_t r_significant = 0, l_significant = 0

        cdef Py_ssize_t [::1] u_row, u_col
        cdef DTYPE_t [::1] u_data
        cdef Py_ssize_t i, j, k, row, col

        cdef tuple L_update, R_update
        cdef Py_ssize_t minibatch_size = len(self.minibatch_rows)
        cdef DTYPE_t acc_gradient

        # check significance
        for i in range(minibatch_size):
            row = self.minibatch_rows[i]
            if unique_rows.get(row, None) is None:
                unique_rows[row] = True
            else:
                continue

            for k in range(self.n_factors):
                acc_gradient = self.L_acc_gradient[row, k]
                if abs(acc_gradient / self.L[row, k]) > threshold:
                    l_u_row[l_significant] = row
                    l_u_col[l_significant] = k
                    l_u_data[l_significant] = acc_gradient
                    l_significant += 1
                    self.L_acc_gradient[row, k] = 0.0

        for j in range(minibatch_size):
            col = self.minibatch_cols[j]
            if unique_cols.get(col, None) is None:
                unique_cols[col] = True
            else:
                continue

            for k in range(self.n_factors):
                acc_gradient = self.R_acc_gradient[col, k]
                if abs(acc_gradient / self.R[col, k]) > threshold:
                    r_u_row[r_significant] = col
                    r_u_col[r_significant] = k
                    r_u_data[r_significant] = acc_gradient
                    r_significant += 1
                    self.R_acc_gradient[col, k] = 0.0

        self.significance = l_significant + r_significant

        # prepare significant updates to be sent
        if l_significant != 0:
            L_update = (np.array(l_u_data[:l_significant]), np.array(l_u_row[:l_significant]), np.array(l_u_col[:l_significant]))
        else:
            L_update = None

        if r_significant != 0:
            R_update = (np.array(r_u_data[:r_significant]), np.array(r_u_row[:r_significant]), np.array(r_u_col[:r_significant]))
        else:
            R_update = None

        if L_update is not None or R_update is not None:
            return L_update, R_update
        else:
            return None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef void aggregate_updates(self, tuple content):
        cdef tuple L_update, R_update
        L_update = content[0]
        R_update = content[1]
        self._aggregate_updates(L_update[0], L_update[1], L_update[2], R_update[0], R_update[1], R_update[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _aggregate_updates(self, DTYPE_t[::1] l_data, Py_ssize_t[::1] l_rows, Py_ssize_t[::1] l_cols,
                                  DTYPE_t[::1] r_data, Py_ssize_t[::1] r_rows, Py_ssize_t[::1] r_cols):
        cdef Py_ssize_t update_size = len(l_data)
        cdef Py_ssize_t l, r

        for l in range(update_size):
            self.L[l_rows[l], l_cols[l]] += l_data[l]

        update_size = len(r_data)
        for r in range(update_size):
            self.R[r_rows[r], r_cols[r]] += r_data[r]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef void aggregate_model(self, list content):
        self._aggregate_model(content[0], content[1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void _aggregate_model(self, DTYPE_t[:, ::1] L_model, DTYPE_t[:, ::1] R_model):
        cdef Py_ssize_t i, j, k

        for i in range(self.L_size):
            for k in range(self.n_factors):
                self.L[i, k] = (self.L[i, k] + L_model[i, k]) / 2.0

        for j in range(self.R_size):
            for k in range(self.n_factors):
                self.R[j, k] = (self.R[j, k] + R_model[j, k]) / 2.0

    @property
    def weights(self):
        return [np.array(self.L), np.array(self.R)]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef DTYPE_t loss_function(self, Py_ssize_t[::1] rows, Py_ssize_t[::1] cols, DTYPE_t[::1] ratings, DTYPE_t[:, ::1] pred, int minibatch_size):
        cdef DTYPE_t[::1] diff = np.zeros(minibatch_size, dtype=DTYPE)
        cdef Py_ssize_t i, j, k, m

        # Kahan summation
        cdef numpy.float_t diff_norm = np.float64(0.)
        cdef numpy.float_t c = np.float64(0.)
        cdef numpy.float_t t, y

        for m in range(minibatch_size):
            i = rows[m]
            j = cols[m]
            for k in range(self.n_factors):
                diff[m] += pred[m][k]
            diff[m] += self.L_bias[i] + self.R_bias[j] - ratings[m]
            diff[m] = diff[m] ** 2

            y = diff[m] - c
            t = diff_norm + y
            c = (t - diff_norm) - y
            diff_norm = t

        return diff_norm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef DTYPE_t[:, ::1] prediction(self, Py_ssize_t[::1] rows, Py_ssize_t[::1] cols, int minibatch_size):
        cdef DTYPE_t[:, ::1] pred = np.zeros((minibatch_size, self.n_factors), dtype=DTYPE)
        cdef Py_ssize_t i, j, k, m

        # compute loss
        for m in range(minibatch_size):
            i = rows[m]
            j = cols[m]
            for k in range(self.n_factors):
                pred[m][k] = self.L[i][k] * self.R[j][k]

        return pred

@cython.cdivision(True)
@cython.initializedcheck(False)
cdef DTYPE_t model_update(DTYPE_t model, DTYPE_t learning_rate, DTYPE_t update):
    return model - learning_rate * update

@cython.initializedcheck(False)
cdef DTYPE_t gradient(DTYPE_t diff, DTYPE_t factor1):
    return 2.0 * diff * factor1

@cython.initializedcheck(False)
cdef DTYPE_t gradient_l2(DTYPE_t diff, DTYPE_t factor1, DTYPE_t reg_param, DTYPE_t factor2):
    return 2.0 * diff * factor1 + reg_param * factor2

@cython.cdivision(True)
@cython.initializedcheck(False)
cdef DTYPE_t velocity(DTYPE_t momentum, DTYPE_t velocity, DTYPE_t gradient, int minibatch_size):
    return (momentum * velocity) + gradient / minibatch_size
