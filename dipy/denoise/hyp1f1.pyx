# cython: boundcheck=False, wraparound=False, cdivision=True, profile=True

from __future__ import division

import math
import cmath
import operator
import numpy as np

cimport numpy as np
cimport cython


cdef double hypsum(int p, int q, list coeffs, double z, int maxterms=6000):

    cdef:
        int k = 0, i
        double s = 1., t = 1., eps = np.finfo(np.float64).eps
        list num = range(p)
        list den = range(p, p + q)

    while True:

        for i in num: t *= (coeffs[i] + k)
        for i in den: t /= (coeffs[i] + k)

        k += 1; t /= k; t *= z; s += t

        if math.fabs(t) < eps:
            return s

        if k > maxterms:
            raise ValueError("Hypergeometric serie did not converge")


cdef double mag(z):
        return np.frexp(math.fabs(z))[1]


cdef expjpi(double x):
    return exp(1j * np.pi * x)


cdef exp(x):
    if type(x) is float:
        return math.exp(x)
    if type(x) is complex:
        return cmath.exp(x)


# cdef power(*args):
#     try:
#         return operator.pow(*(float(x) for x in args))
#     except (TypeError, ValueError):
#         return operator.pow(*(complex(x) for x in args))


cdef isnpint(x):
    if type(x) is complex:
        if x.imag:
            return False
        x = x.real
    return x <= 0.0 and round(x) == x


cdef nint_distance(z):
    cdef int n = round(z.real)

    if n == z:
        return n, -np.inf
    return n, mag(math.fabs(z-n))


cdef _check_need_perturb(terms, bint discard_known_zeros):

    cdef:
        bint perturb = False
        bint have_singular_nongamma_weight = False
        int n, i, term_index, data_index
        # double w_s, c_s, alpha_s, beta_s, a_s, b_s, z, d, x
        int [::1] discard = np.array([], dtype=np.int32)
        int [::1] pole_count = np.zeros(3, dtype=np.int32)

    for term_index, term in enumerate(terms):
        w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term
        # Avoid division by zero in leading factors (TODO:
        # also check for near division by zero?)
        for k, w in enumerate(w_s):
            if not w:
                if np.real(c_s[k]) <= 0 and c_s[k]:
                    perturb = True
                    have_singular_nongamma_weight = True

        # Check for gamma and series poles and near-poles
        for data_index, data in enumerate([alpha_s, beta_s, b_s]):
            for i, x in enumerate(data):
                n, d = nint_distance(x)
                # Poles
                if n > 0:
                    continue
                if d == -np.inf:
                    # OK if we have a polynomial
                    # ------------------------------
                    if data_index == 2:
                        for u in a_s:
                            if isnpint(u) and u >= int(n):
                                break
                    else:
                        pole_count[data_index] += 1
        if (discard_known_zeros and
            pole_count[1] > pole_count[0] + pole_count[2] and
            not have_singular_nongamma_weight):
            discard.append(term_index)
        elif np.sum(pole_count):
            perturb = True
    return perturb, discard


cdef  double hypercomb(function, double [::1] params, bint discard_known_zeros=True):
    cdef:
        bint perturb
        int discard, k, term_index
        double [::1] evaluated_terms = np.array([], dtype=np.int32)
        double h = 3.0517578125e-05
        # double sumvalue, w_s, c_s, alpha_s, beta_s, a_s, b_s, z, term_data, i

    terms = function(*params)
    perturb, discard =  _check_need_perturb(terms, discard_known_zeros)
    if perturb:
        for k in range(len(params)):
            params[k] += h
            # Heuristically ensure that the perturbations
            # are "independent" so that two perturbations
            # don't accidentally cancel each other out
            # in a subtraction.
            h += h / ( k + 1)
        terms = function(*params)
    if discard_known_zeros:
        terms = [term for (i, term) in enumerate(terms) if i not in discard]
    if not terms:
        return 0.

    for term_index, term_data in enumerate(terms):
        w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term_data
        # Always hyp2f0
        assert len(a_s) == 2
        assert len(b_s) == 0
        v = np.prod([hypsum(2, 0, a_s, z)] + \
            [math.gamma(a) for a in alpha_s] + \
            [1. / math.gamma(b) for b in beta_s] + \
            [w**c for (w,c) in zip(w_s,c_s)])
        evaluated_terms.append(v)

    if len(terms) == 1 and (not perturb):
        return evaluated_terms[0]

    sumvalue = np.sum(evaluated_terms)
    return sumvalue


# cdef double _hyp1f1(double a, double b, double z):
#     cdef:
#         double magz, rz, v
#         double [::1] arr = np.array([a,b], dtype=np.float64)

#     magz = mag(z)

#     if magz >= 7:
#         try:

#             def h(a,b):
#                 E = expjpi(a)
#                 rz = 1./z
#                 T1 = ([E,z], [1,-a], [b], [b-a], [a, 1+a-b], [], -rz)
#                 T2 = ([exp(z),z], [1,a-b], [b], [a], [b-a, 1-a], [], rz)
#                 return T1, T2

#             v = hypercomb(h, arr)
#             return np.real(v)

#         except ValueError("Hypergeometric serie did not converge"):
#             pass

#     return hypsum(1, 1, [a, b], z)


# def  hyp1f1(a, b, z):
#     return _hyp1f1(a, b, z)

def hyp1f1(a, b, z):

    magz = mag(z)

    if magz >= 7:
        try:

            def h(a,b):
                E = expjpi(a)
                rz = 1./z
                T1 = ([E,z], [1,-a], [b], [b-a], [a, 1+a-b], [], -rz)
                T2 = ([exp(z),z], [1,a-b], [b], [a], [b-a, 1-a], [], rz)
                return T1, T2

            v = hypercomb(h, np.array([a,b], dtype=np.float64))
            return np.real(v)

        except ValueError("Hypergeometric serie did not converge"):
            pass

    return hypsum(1, 1, [a, b], z)
