import cython
from lithops.libs.model cimport DTYPE_t

cdef class CModel:
    cdef DTYPE_t threshold
    cdef public int significance

    # Allow dynamic attributes
    cdef dict __dict__

    def __cinit__(self, **kwargs):
        # Gaia implementation details
        self.threshold = kwargs['asp_threshold']
        self.significance = 0


class Model:
    def __init__(self, cython_model):
        self.model = cython_model

    def step(self, epoch, step, minibatch):
        return self.model.step(epoch, step, minibatch)

    def aggregate_updates(self, content):
        self.model.aggregate_updates(content)

    def aggregate_model(self, content):
        self.model.aggregate_model(content)

    def get_significant_updates(self, step):
        return self.model.get_significant_updates(step)

    def get_weights(self):
        return self.model.weights

    @property
    def significance(self):
        return self.model.significance
