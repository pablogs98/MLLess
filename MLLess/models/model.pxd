cimport numpy

ctypedef numpy.float32_t DTYPE_t

cdef class CModel:
    cdef float threshold
    cdef public int significance

    # Allow dynamic attributes
    cdef dict __dict__