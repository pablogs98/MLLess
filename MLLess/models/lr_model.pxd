from lithops.libs.model import CModel
cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t

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