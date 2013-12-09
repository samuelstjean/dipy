from __future__ import division
import numpy as np

from scipy.special import iv, factorial

cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def _marcumq(float a, float b, int M, float eps=10**-10):

    if b == 0:
        return 1

    cdef np.ndarray ka = np.arange(M, dtype=np.int)

    if a == 0:
        return np.exp(-b**2/2) * np.sum(b**(2*ka) / (2**ka * factorial(ka)))

    cdef float z = a * b
    cdef int k
    cdef int s
    cdef int c
    cdef float x
    cdef float d
    cdef float S
    cdef float t

    if a < b:

        s = 1
        c = 0
        x = a/b
        d = x
        S = iv(0, z) * np.exp(-z)

        for k in range(1, M):
            t = (d + 1/d) * iv(k, z) * np.exp(-z)
            S += t
            d *= x

        k += 1

    else:
        s = -1
        c = 1
        x = b/a
        k = M
        d = x**M
        S = 0
        t = 1

    cdef bint condition = True

    while (condition):

        t = d * iv(k, z) * np.exp(-z)
        S += t
        d *= x
        k += 1

        condition = np.abs(t/S) > eps

    return c + s * np.exp(-0.5 * (a-b)**2) * S
