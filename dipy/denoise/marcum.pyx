from __future__ import division
import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
def _marcumq(float a, float b, int M, float eps=10**-10):

    if b == 0:
        return 1

    if a == 0:

        cdef int k = np.arange(M)

        return np.exp(-b**2/2) * np.sum(b**(2*k) / (2**k * factorial(k)))

    cdef float z = a * b
    cdef int k = 0

    if a < b:

        cdef int s = 1
        cdef int c = 0
        cdef float x = a/b
        cdef float d = x
        cdef float S = iv(0, z) * np.exp(-z)

        for k in range(1, M):
            cdef float t = (d + 1/d) * iv(k, z) * np.exp(-z)
            cdef float S += t
            cdef float d *= x

        k += 1

    else:
        cdef int s = -1
        cdef int c = 1
        cdef float x = b/a
        cdef int k = M
        cdef float d = x**M
        cdef float S = 0
        cdef float t = 1

    cdef bool condition = True

    while (condition):

        t = d * iv(np.abs(k), z) * np.exp(-z)
        S += t
        d *= x
        k += 1

        condition = np.abs(t/S) > eps


    return c + s * np.exp(-0.5 * (a-b)**2) * S
