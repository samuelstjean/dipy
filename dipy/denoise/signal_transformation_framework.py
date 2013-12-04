from __future__ import division, print_function

import numpy as np

from scipy.special import erfinv, hyp1f1, iv
from scipy.misc import factorial, factorial2
from scipy.integrate import quad


def _inv_cdf_gauss(y, eta, sigma):
    return eta + sigma * np.sqrt(2) * erfinv(2*y - 1)


def _cdf_nchi(alpha, eta, sigma, N):
    return quad(_pdf_nchi, 0, alpha, args=(eta, sigma, N), limit=250)[0]


def _pdf_nchi(m, eta, sigma, N):
    return m**N/(sigma**2 * eta**(N-1)) * np.exp((m**2 + eta**2)/(-2*sigma**2)) * iv(N-1, m*eta/sigma**2)


def _beta(N):
    return np.sqrt(np.pi/2) * (factorial2(2*N-1)/(2**(N-1) * factorial(N-1)))


def _xi(eta, sigma, N):
    return 2*N + eta**2/sigma**2 - (_beta(N) * hyp1f1(-0.5, N, -eta**2/(2*sigma**2)))**2


def _fixed_point_g(eta, m, sigma, N):
    return np.sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)


def _fixed_point_k(eta, m, sigma, N):

    fpg = _fixed_point_g(eta, m, sigma, N)
    num =  fpg * (fpg - eta)

    denom = eta * (1 - ((_beta(N)**2)/(2*N)) *
            hyp1f1(-0.5, N, -eta**2/(2*sigma**2)) *
            hyp1f1(0.5, N+1, -eta**2/(2*sigma**2))) - fpg

    return eta - num / denom


def chi_to_gauss(m, eta, sigma, N, alpha=0.0005):

    vec_cdf_nchi = np.vectorize(_cdf_nchi)
    cdf = vec_cdf_nchi(m, eta, sigma, N)

    # Find outliers and clip them to confidence interval limits
    np.clip(cdf, alpha/2, 1 - alpha/2, out=cdf)

    return _inv_cdf_gauss(cdf, eta, sigma)


def fixed_point_finder(m, sigma, N, max_iter=500, eps=10**-10):

    delta = _beta(N) * sigma - m

    if delta == 0:
        return 0

    if delta > 0:
        m = _beta(N) * sigma + delta

    t0 = m
    t1 = _fixed_point_k(t0, m, sigma, N)
    n_iter = 0

    while(np.abs(t0 - t1) > eps):
        t0 = t1
        t1 = _fixed_point_k(t0, m, sigma, N)
        n_iter += 1

        if n_iter > max_iter:
            break

    if delta > 0:
        return -t1

    return t1
