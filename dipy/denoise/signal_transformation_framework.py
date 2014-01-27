from __future__ import division, print_function

import numpy as np

from scipy.special import erfinv, hyp1f1, iv, ive, gammainccinv
from scipy.misc import factorial, factorial2
from scipy.stats import mode
from scipy.linalg import svd

from dipy.denoise.noise_field import _noise_field

#from dipy.core.ndindex import ndindex
#import mpmath as mp
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
    # Real formula is below, but LUT is faster since N is fixed at the beginning
    # Other values of N can be easily added by simply using the last line
    # and then adding them to values.

    values = {1: 1.25331413732,
              2: 1.87997120597,
              4: 2.74162467538,
              6: 3.39276053578,
              8: 3.93802562189,
              12: 4.84822789808,
              16: 5.61283938922,
              20: 6.28515420794,
              24: 6.89221524065,
              36: 8.45587062694,
              48: 9.77247710766,
              64: 11.2916332015}

    return values[N]

    #return np.sqrt(np.pi/2) * (factorial2(2*N-1)/(2**(N-1) * factorial(N-1)))


def _xi(eta, sigma, N):
    ##print(np.sum(np.isnan(eta**2)), np.sum(np.isnan(hyp1f1(-0.5, N, -eta**2/(2*sigma**2)))))
    return 2*N + eta**2/sigma**2 - (_beta(N) * hyp1f1(-0.5, N, -eta**2/(2*sigma**2)))**2
    # out = 2*N + eta**2/sigma**2 - (_beta(N) * hyp1f1(-0.5, N, -eta**2/(2*sigma**2)))**2

    # print(np.sum(np.isnan(out)), np.sum(np.isinf(out)), "in")
    # idx = np.logical_or(np.isnan(out), np.isinf(out))
    # out[idx] = eta[idx]


    # #hyp = np.frompyfunc(mp.hyp1f1, 3, 1)
    # #out[idx] = np.array(hyp(-0.5, N, -eta[idx]**2/(2*sigma**2)), dtype=np.float64)
    # #out[idx] = 2*N + eta[idx]**2/sigma**2 - (_beta(N) * out[idx])**2

    # #_dtype_object = np.dtype('object')
    # #_cfunc_mpf = np.vectorize(mp.mpf, otypes=(_dtype_object,))
    # #func = _cfunc_mpf
    # #tmp = eta[idx]
    # #res = func(tmp)
    # #res2 = 2*N + tmp**2/sigma**2 - (_beta(N) * mp.hyp1f1(-0.5, N, -tmp**2/(2*sigma**2)))**2
    # #out[idx] = res2
    # #print(np.sum(np.isnan(res2)), np.sum(np.isinf(res2)), "out")
    # print(np.sum(np.isnan(out)), np.sum(np.isinf(out)), "out")
    # return out

    # hyp = np.frompyfunc(mp.hyp1f1, 3, 1)
    # out = 2*N + eta**2/sigma**2 - (_beta(N) * hyp1f1(-0.5, N, -eta**2/(2*sigma**2)))**2
    # out[np.isnan(out)] = np.array(hyp(-0.5, N, -eta[np.isnan(out)]**2/(2*sigma**2)), dtype=np.float64)
    # print(np.sum(np.isnan(out)))
    # print(np.sum(np.isnan(2*N + eta**2/sigma**2 - (_beta(N) * out)**2)))
    # return 2*N + eta**2/sigma**2 - (_beta(N) * out)**2

    # out = np.zeros_like(eta)
    # div = 2*sigma**2

    # for idx in np.ndindex(eta.shape):
    #     out[idx] = hyp1f1(-0.5, N, -eta[idx]**2/div)

    # return 2*N + eta**2/sigma**2 - (_beta(N) * out)**2


def _fixed_point_g(eta, m, sigma, N):
    #print(np.sum(np.isnan(_xi(eta, sigma, N))))
    #print(np.sum(np.sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2) < 0))
    #print(np.sum(np.isinf(_xi(eta, sigma, N))), np.sum(np.isinf(np.sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2))))
    return np.sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)


def _fixed_point_k(eta, m, sigma, N):

    fpg = _fixed_point_g(eta, m, sigma, N)
    num = fpg * (fpg - eta)

    denom = eta * (1 - ((_beta(N)**2)/(2*N)) *
                   hyp1f1(-0.5, N, -eta**2/(2*sigma**2)) *
                   hyp1f1(0.5, N+1, -eta**2/(2*sigma**2))) - fpg

    #print(np.max(eta), np.min(eta), np.min(-eta**2/(2*sigma**2)), np.max(-eta**2/(2*sigma**2)))
    #idx = np.logical_or(np.isnan(denom), np.isinf(denom))
    #out[idx] = eta[idx]
    #hyp = np.frompyfunc(mp.hyp1f1, 3, 1)
    #denom[idx] = np.array(eta[idx] * (1 - ((_beta(N)**2)/(2*N)) *
    #                      hyp(-0.5, N, -eta[idx]**2/(2*sigma**2)) *
    #                      hyp(0.5, N+1, -eta[idx]**2/(2*sigma**2))) - fpg[idx], dtype=np.float64)

    #print(np.sum(np.isnan(eta - num / denom)), np.sum(np.isinf(eta - num / denom)), "fpk")
    #print(np.sum(np.isnan(eta)), np.sum(np.isnan(num)), np.sum(np.isnan(denom)))
    return eta - num / denom


def _marcumq(a, b, M, eps=10**-12):

    if np.all(b == 0):
        return np.ones_like(b)

    if np.all(a == 0):
        #k = np.arange(M)
        #return np.exp(-b**2/2) * np.sum(b**(2*k) / (2**k * factorial(k)))

        temp = 0
        for k in range(M):
            temp += b**(2*k) / (2**k * factorial(k))

        return np.exp(-b**2/2) * temp

    z = a*b
    #expz = np.exp(-z)
    k = 0
    #print(np.sum(np.isnan(expz)), np.sum(np.isinf(expz)))
    #print(np.sum(np.isnan(iv(0, z))), np.sum(np.isinf(iv(0, z))))
    #print(np.min(expz), np.min(z), np.min(a), np.min(b), np.max(z), np.max(a), np.max(b))
    if np.all(a < b):

        s = 1
        c = 0
        x = a/b
        d = x
        S = ive(0, z)

        for k in range(1, M):

            S += (d + 1/d) * ive(k, z)
            d = d * x

        k += 1

    else:

        s = -1
        c = 1
        x = b/a
        k = M
        d = x**M
        S = np.zeros_like(z, dtype=np.float64)

    cond = True  # np.ones_like(z, dtype=np.bool)

    while np.all(cond):

        t = d * ive(k, z)
        S += t
        d = d * x
        k += 1

        cond = np.abs(t/S) > eps

    return c + s * np.exp(-0.5 * (a-b)**2) * S


#vec_cdf_nchi = np.vectorize(_cdf_nchi, otypes=["float64"], cache=True)
#class ncx(stats.rv_continuous):
#    def _pdf(self, m, eta, sigma, N):
#        return m**N/(sigma**2 * eta**(N-1)) * np.exp((m**2 + eta**2)/(-2*sigma**2)) * iv(N-1, m*eta/sigma**2)


def estimate_sigma_grappa(data, grappa_kernel_W=None, cov_matrix=None, L=12, r=2, n=3):
    """Estimation of the standard deviation of noise in parallel MRI.

    data : Data to estimate the noise variance from

    theta : LxL covariance matrix of GRAPPA reconstruction weights (eq. 15-16)

    L : Number of channel in the receiver coils

    r : GRAPPA acceleration factor

    n : Radius of the neighboorhood used to estimate m2
    """

    if grappa_kernel_W is None and cov_matrix is None:
        theta = np.eye(L)
    else:
        if not grappa_kernel_W.shape == (L, L):
            raise ValueError("GRAPPA kernel matrix must be of shape %ix%i, \
                but is of shape" % L, grappa_kernel_W.shape)

        if not cov_matrix.shape == (L, L):
            raise ValueError("Covariance matrix must be of shape %ix%i, \
                but is of shape" % L, cov_matrix.shape)

        theta = np.dot(np.dot(grappa_kernel_W, cov_matrix), grappa_kernel_W.T)

    m2 = _estimate_m2(data, n)

    #m2 = _expected_m2(sigma2n, trace_theta)
    #sigma2n = _sigma2n(m2, trace_theta)

    return np.sqrt(_sigma2_eff(theta, m2, L))


def _estimate_m2(data, n):

    #padded = np.pad(data, (n, n), mode='reflect')
    #m2 = np.zeros_like(padded, dtype=np.float64)

    # pad, mais on veut juste padder 3D en realite et utiliser toutes les DWIs
    padded = np.pad(data[..., data.shape[-1]:-2*n], (n, n), mode='reflect')
    m2 = np.zeros((padded.shape, data.shape[-1]), dtype=np.float64)

    for idx in (data.shape[-1]):

        a = np.array(idx) + (n - 1)
        b = a + 2 * n + 1
        m2[idx] = np.mean(padded[a:b,
                                 a:b,
                                 a:b, :]**2)
        print(idx,a,b)

    return m2[n:-n, n:-n, n:-n, :]


#def _L_eff(L, A, sigma2):
# Eq. 17 p.4


#def _sigma2_eff(phi, sigma2b, sigma2s):
# Eq. 33 p.4
#    return phi * sigma2b + (1 - phi) * sigma2s


def _sigma2_eff(theta, m2, L):
# Eq. 35 p. 6
    trace_theta = np.trace(theta)
    sigma2n = 0.5 * mode(m2/trace_theta, axis=None)
    print(sigma2n)
    sigma2n = sigma2n[0]
    tts = trace_theta * sigma2n

    return sigma2n * ((tts)/(m2 - tts) * (np.abs(np.sum(theta))/L)
        + ((1 - tts)/(m2 - tts)) * np.sum(theta**2)/trace_theta)

#def _SNR2(A, L, sigma2):


#def _phin(sigma2n, trace_theta, m2):
# Eq. 34 p.6
#    tts = trace_theta * sigma2n
#    return tts / (m2 - tts)


#def _expected_m2(sigma2n, trace_theta):
# Eq. 30 p. 5
#    return 2 * sigma2n * trace_theta


#def _sigma2n(m2, trace_theta):
# Eq. 32 p. 5
#    return 0.5 * mode(m2/trace_theta)


def chi_to_gauss(m, eta, sigma, N=12, alpha=0.0005):

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
    #print(np.sum(cdf < alpha/2), np.sum(cdf > 1 - alpha/2), np.sum(np.logical_or(alpha/2 < cdf, cdf < 1 - alpha/2)))
    #print(np.sum(np.isnan(cdf)), np.sum(np.isinf(cdf)))
    #cdf[np.isnan(cdf)] = 0
    #cdf[np.isinf(cdf)] = 1
    np.clip(cdf, alpha/2, 1 - alpha/2, out=cdf)

    return _inv_cdf_gauss(cdf, eta, sigma)


def fixed_point_finder(m, sigma, N=12, max_iter=500, eps=10**-12):

    m = m.astype('float64')
    delta = _beta(N) * sigma - m
    out = np.zeros_like(delta)
    t0 = np.zeros_like(delta)
    t1 = np.zeros_like(delta)
    ind = np.zeros_like(delta, dtype=np.bool)

    for idx in [delta < 0, delta > 0]:

        if np.all(delta[idx] > 0):
            print ("delta > 0", np.sum(idx))
        elif np.all(delta[idx] < 0):
            print ("delta < 0", np.sum(idx))
        else:
            print("oups")

        #if delta == 0:
        #    return 0

        #if np.all(delta[idx] != 0):

        if np.all(delta[idx] > 0):
            m[idx] = _beta(N) * sigma + delta[idx]

        t0[idx] = m[idx]
        t1[idx] = _fixed_point_k(t0[idx], m[idx], sigma, N)
        #t0 = m[idx]
        #t1 = _fixed_point_k(t0, m[idx], sigma, N)

        n_iter = 0
        #print(t0.shape, t1.shape, idx.shape)
        ind[idx] = np.abs(t0[idx] - t1[idx]) > eps
        #print(np.sum(np.isnan(t1[idx])), "t1")
        #print(np.sum(np.isnan(delta)), "delta")
        #while np.any(np.abs(t0 - t1) > eps):
        while np.any(ind):

            #t0 = t1
            #t1 = _fixed_point_k(t0, m[idx], sigma, N)
            #n_iter += 1

            t0[ind] = t1[ind]
            t1[ind] = _fixed_point_k(t0[ind], m[ind], sigma, N)
            n_iter += 1
            ind[idx] = np.abs(t0[idx] - t1[idx]) > eps

            if n_iter > max_iter:
                break

        if np.all(delta[idx] > 0):
            out[idx] = -t1[idx]
            #return -t1
        if np.all(delta[idx] < 0):
            out[idx] = t1[idx]

    return out
    #return t1


def piesno(data, N=12, alpha=0.01, l=100, itermax=100, eps=10**-12):
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
        The number of phase array coils of the mr scanner.

    alpha : float
        Probabilistic estimation threshold for the gamma function.

    l : int
        number of initial estimates for sigma to try.

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
    denom = np.sqrt(2 * _inv_nchi_cdf(N, 1, 1/2))
    m = np.median(data) / denom

    phi = np.arange(1, l + 1) * m/l
    K = data.shape[-1]
    sum_m2 = np.sum(data**2, axis=-1)

    sigma = np.zeros_like(phi)

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


def _chi(SNR):
    return 2 + SNR**2 - np.pi/8 * np.exp(-SNR**2/2) * ((2 + SNR**2) * iv(0, SNR**2/4) + SNR**2 * iv(1, SNR**2/4))**2


def estimate_noise_field(data, radius=1):
    """Estimates the noise field using B0s or DWIs and PCA."""

    b0s = data[..., 0]
    dwis = data[..., 1:]
    dwis = np.reshape(dwis, (dwis.shape[-1], -1))
    print(dwis.shape)

    mean = np.mean(dwis, axis=1, keepdims=True)
    dwis -= mean
    sigma = np.dot(dwis, dwis.T) / (dwis.shape[-1] - 1)
    #sigma = dwis.T / np.sqrt(dwis.shape[-1] - 1)
    #sub = mean/np.sqrt(dwis.shape[-1] - 1)
    #dwis -= sub
    #print(mean.shape, sigma.shape)
    U, s, Vt = svd(sigma)
    # noise = np.dot(U[:, -1], dwis).reshape(data.shape[:-1], -1)
    print(U.shape, Vt.shape, np.sum(np.abs(U-Vt.T)))








    # #Compute mean and covariance
    # dwis = dwis.T
    # mean = dwis.mean(axis=0, keepdims=True)
    # data_cov = np.cov(dwis, rowvar=0)
    # #Add a small constant on the diagonal, to regularize
    # #data_cov += np.diag(regularizer*np.ones(input_size))
    # #Compute the principal components
    # dwis -= mean
    # data_cov = np.cov(dwis, rowvar=1) #np.dot(dwis, dwis.T) / (dwis.shape[1] - 1)
    # w,v = np.linalg.eigh(data_cov)
    # s = (-w).argsort()
    # w = w[s]
    # v = v[:,s]

    # #Convert arrays to garrays to use the GPU for the whitening process
    # #projection = gpu.garray(v)
    # #scaling = gpu.garray(1./np.sqrt(w))
    # #transform = scaling.reshape((1,-1))*projection

    # #ZCA whitening
    # #print((dwis-mean).shape, v.shape)
    # return np.dot(v, np.dot(v, dwis-mean)).reshape(data.shape[:-1] + (-1,))



    #noise_comp = np.zeros_like(U)
    #noise_comp[..., -1] = U[..., -1]
    #noise_comp = U[-1:, ...]
    noise_comp = U[..., -1:]#[..., None]
   # noise_comp = Vt[...,-1:].T
    print(noise_comp.shape)
    #noise_comp = U
    #recon = np.dot(noise_comp.T, np.dot(noise_comp, dwis)) #+ mean
    #recon = np.dot(noise_comp, np.dot(noise_comp.T, dwis)) + mean
    recon = np.dot(noise_comp.T, dwis) + mean
    #return dwis.reshape(data.shape[:-1] + (-1,))
    #dwis += mean

    #noise_comp = np.zeros_like(U)
    #noise_comp[..., -1:] = U[..., -1:]
    #recon = np.dot(noise_comp, np.dot(noise_comp.T, dwis)) + mean

    noise_field = recon.reshape(data.shape[:-1] + (-1,))





    return _noise_field(noise_field, radius)
    #s_noise = np.zeros_like(s)
    #s_noise[-1] = s[-1]
    #noise = np.dot(U * s_noise, Vt) += sub