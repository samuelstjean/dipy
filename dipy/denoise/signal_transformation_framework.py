from __future__ import division, print_function

import numpy as np

from scipy.special import erfinv, hyp1f1, iv, gammainccinv
from scipy.misc import factorial, factorial2
#from scipy.integrate import quad, romberg, romb
#from scipy import stats

#from dipy.core.ndindex import ndindex


def _inv_cdf_gauss(y, eta, sigma):
    return eta + sigma * np.sqrt(2) * erfinv(2*y - 1)


def _inv_nchi_cdf(N, K, alpha):
    """Inverse CDF for the noncentral chi distribution
    See [1]_ p.3 section 2.3"""
    return gammainccinv(N * K, 1 - alpha) / K


# def _cdf_nchi(alpha, eta, sigma, N):

#     #return 1 - _marcumq(eta/sigma, alpha/sigma, N)
#     #print(alpha.shape,eta.shape,sigma.shape)
#     out = np.zeros_like(alpha)
#     for idx in range(alpha.size):
#         out[idx] = romberg(_pdf_nchi, 0, alpha[idx], args=(eta, sigma, N))
#     return out
#     #return romberg(_pdf_nchi, 0, alpha, args=(eta, sigma, N))#[0]#, limit=250)[0]
#     #sample = np.linspace(0, alpha)
#     #return romb(y)

# def _pdf_nchi(m, eta, sigma, N):
#     return m**N/(sigma**2 * eta**(N-1)) * np.exp((m**2 + eta**2)/(-2*sigma**2)) * iv(N-1, m*eta/sigma**2)


def _beta(N):
    return np.sqrt(np.pi/2) * (factorial2(2*N-1)/(2**(N-1) * factorial(N-1)))


def _xi(eta, sigma, N):
    return 2*N + eta**2/sigma**2 - (_beta(N) * hyp1f1(-0.5, N, -eta**2/(2*sigma**2)))**2


def _fixed_point_g(eta, m, sigma, N):
    return np.sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)


def _fixed_point_k(eta, m, sigma, N):

    fpg = _fixed_point_g(eta, m, sigma, N)
    num = fpg * (fpg - eta)

    denom = eta * (1 - ((_beta(N)**2)/(2*N)) *
                   hyp1f1(-0.5, N, -eta**2/(2*sigma**2)) *
                   hyp1f1(0.5, N+1, -eta**2/(2*sigma**2))) - fpg

    return eta - num / denom


def _marcumq(a, b, M, eps=10**-16):

    if np.all(b == 0):
        return np.ones_like(b)

    if np.all(a == 0):

        k = np.arange(M)
        return np.exp(-b**2/2) * np.sum(b**(2*k) / (2**k * factorial(k)))

    z = a * b
    expz = np.exp(-z)
    k = 0

    if np.all(a < b):

        s = 1
        c = 0
        x = a/b
        d = x
        S = iv(0, z) * expz

        for k in range(1, M):

            S += (d + 1/d) * iv(k, z) * expz
            d = d * x

        k += 1

    else:
        s = -1
        c = 1
        x = b/a
        k = M
        d = x**M
        S = 0

    condition = True

    while np.all(condition):

        t = d * iv(k, z) * expz
        S += t
        d = d * x
        k += 1

        condition = np.abs(t/S) > eps

    return c + s * np.exp(-0.5 * (a-b)**2) * S


#vec_cdf_nchi = np.vectorize(_cdf_nchi, otypes=["float64"], cache=True)
#class ncx(stats.rv_continuous):
#    def _pdf(self, m, eta, sigma, N):
#        return m**N/(sigma**2 * eta**(N-1)) * np.exp((m**2 + eta**2)/(-2*sigma**2)) * iv(N-1, m*eta/sigma**2)


def chi_to_gauss(m, eta, sigma, N, alpha=0.0005):

    #cdf = np.clip(1 - _marcumq(eta/sigma, m/sigma, N), alpha/2, 1 - alpha/2)

    # m = np.array(m)
    cdf = np.zeros_like(m, dtype=np.float64)

    # # eta = 0 => cdf is zero
    # if eta > 0:
    for idx in [np.logical_and(eta/sigma < m/sigma, eta > 0),
                np.logical_and(eta/sigma >= m/sigma, m > 0),
                m == 0,
                eta == 0]:

        if cdf[idx].size > 0:
            cdf[idx] = np.array(1 - _marcumq(eta[idx]/sigma, m[idx]/sigma, N))

    # Find outliers and clip them to the confidence interval limits
    np.clip(cdf, alpha/2, 1 - alpha/2, out=cdf)

    return _inv_cdf_gauss(cdf, eta, sigma)


def fixed_point_finder(m, sigma, N, max_iter=500, eps=10**-10):

    delta = _beta(N) * sigma - m
    out = np.zeros_like(delta)

    for idx in [delta < 0, delta > 0, delta == 0]:

        #if delta == 0:
        #    return 0

        if np.all(delta[idx] == 0):
            out[idx] = 0

        else:

            if np.all(delta[idx] > 0):
                m[idx] = _beta(N) * sigma + delta[idx]

            t0 = m[idx]
            t1 = _fixed_point_k(t0, m[idx], sigma, N)
            n_iter = 0

            while np.any(np.abs(t0 - t1) > eps):

                t0 = t1
                t1 = _fixed_point_k(t0, m[idx], sigma, N)
                n_iter += 1

                if n_iter > max_iter:
                    break

            if np.all(delta[idx] > 0):
                out[idx] = -t1
                #return -t1
            if np.all(delta[idx] < 0):
                out[idx] = t1

    return out
    #return t1


def piesno(data, N=12, alpha=0.01, l=100, itermax=100, eps=10**-10):
    """
    A routine for finding the underlying gaussian distribution standard
    deviation from magnitude signals.

    This is a re-implementation of [1]_ and the second step in the
    stabilisation framework of [2]_.

    Parameters
    -----------

    data : numpy array
        The magnitude signals to analyse. The last dimension must contain the
        same realisation of the volume, such as dMRI or fMRI data.

    N : int
        The number of phase array coils of the mr scanner

    alpha : float
        Probabilistic estimation threshold for the gamma function.

    l : int
        number of initial estimates for sigma to try

    itermax : int
        Maximum number of iterations to execute if convergence
        is not reached.

    eps : float
        Tolerance for the convergence criterion. Convergence is
        reached if two subsequent estimates are smaller than eps.

    References
    ------------

    .. [1]. Koay CG, Ozarslan E and Pierpaoli C.
    "Probabilistic Identification and Estimation of Noise (PIESNO):
    A self-consistent approach and its applications in MRI."
    Journal of Magnetic Resonance 2009; 199: 94-103.

    .. [2] Koay CG, Ozarslan E and Basser PJ.
    "A signal transformational framework for breaking the noise floor
    and its applications in MRI."
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """

    # Initial estimation of sigma
    m = np.median(data)

    phi = np.arange(1, l + 1) * m/l
    K = data.shape[-1]
    sum_m2 = np.sum(data**2, axis=-1)

    sigma = np.zeros_like(phi)
    denom = np.sqrt(2 * _inv_nchi_cdf(N, 1, 1/2))

    lambda_minus = _inv_nchi_cdf(N, K, alpha/2)
    lambda_plus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    pos = 0
    max_length_omega = 0

    for num, sig in enumerate(phi):

        sig_prev = 0
        omega_size = 1

        for n in range(itermax):

            if np.abs(sig - sig_prev) < eps:
                break

            s = sum_m2 / (2*K*sig**2)
            idx = np.logical_and(lambda_minus <= s, s <= lambda_plus)
            omega = data[idx, :]

            # If no point meets the criterion, exit
            if omega.size == 0:
                omega_size = 0
                break

            sig_prev = sig
            sig = np.median(omega) / denom
            omega_size = omega.size/K

        # Remember the biggest omega array as giving the optimal
        # sigma amongst all initial estimates from phi
        if omega_size > max_length_omega:
            pos, max_length_omega = num, omega_size

        sigma[num] = sig

    return sigma[pos]
