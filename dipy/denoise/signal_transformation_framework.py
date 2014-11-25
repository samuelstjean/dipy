from __future__ import division, print_function

from time import time
from copy import copy

import numpy as np

from scipy.special import erfinv, ive, gammainccinv # hyp1f1,
from scipy.misc import factorial, factorial2
from scipy.stats import mode
from scipy.linalg import svd

from dipy.core.ndindex import ndindex

from scilpy.denoising.hyp1f1 import hyp1f1

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


def _optimal_quantile(N):
    """Returns the optimal quantile order alpha for a known N"""

    values = {1: 0.79681213002002,
              2: 0.7306303027491917,
              4: 0.6721952960782169,
              8: 0.6254030432343569,
             16: 0.5900487123737876,
             32: 0.5641772300866416,
             64: 0.5455611840489607,
            128: 0.5322811923303339}

    if N in values:
        return values[N]
    else:
        return 0.5


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


def _marcumq_matlab(a, b, M, eps=1e-7):

    # if np.all(np.abs(b) < eps):
    #     return np.ones_like(b)

    # if np.all(np.abs(a) < eps):
    #     temp = 0.
    #     for k in range(M):
    #         temp += b**(2*k) / (2**k * factorial(k))

    #     return np.exp(-b**2/2) * temp

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    aa = 0.5 * a**2
    bb = 0.5 * b**2
    d = np.exp(-aa)
    h = copy(d)
    f = (bb**M) * np.exp(-bb) / factorial(M)
    f_err = np.exp(-bb)
    errbnd = 1 - f_err
 #   errbnd[errbnd == 1] = 0.  # exp too negative
    k = 1
    delta = f * h
    S = copy(delta)
    j = np.array((errbnd > 4*eps), dtype=np.bool) #& ((1-S) > 8*eps), dtype=np.bool)

    while np.any(j):# | k <= m:
        d[j] = aa[j] * d[j]/k
        # d[j] *= aa[j]/k
        h[j] += d[j]
        f[j] = bb[j] * f[j] / (k + M)
        # f[j] = bb[j] / (k + M)
        delta[j] = f[j] * h[j]
        S[j] += delta[j]
        f_err[j] *= bb[j] / k
        errbnd[j] -= f_err[j]
        j = (errbnd > 4*eps) # & ((1 - S) > 8*eps)
        k += 1
        # print(k)
        # print(delta)
        # print(S)
        # print(errbnd)
        # print(eps*(1-S))
        # print(errbnd > 4*eps)
        # print((1 - S) > 8*eps)
        # print(np.sum(errbnd > 4*eps))
        # print(errbnd[j])
        # print(f_err[j])
        # if (k > 100000):
        #     j = j & np.any((delta > eps*(1-S)))

    return 1 - S


def _marcumq_octave(a, b, M, eps=1e-7):

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    if np.all(np.abs(b) < eps):
        return np.ones_like(b)

    if np.all(np.abs(a) < eps):
        temp = 0.
        for k in range(M):
            temp += b**(2*k) / (2**k * factorial(k))

        return np.exp(-b**2/2) * temp

    z = a * b
    k = 0

    if np.all(a < b):

        s = 1
        c = 0
        x = a / b
        d = copy(x)
        S = ive(0, z)

        for k in range(1, M):

            S += (d + 1/d) * ive(k, z)
            d *= x

        k += 1

    else:

        s = -1
        c = 1
        x = b / a
        k = M
        d = x**M
        S = np.zeros_like(z, dtype=np.float64)

    cond = np.ones_like(z, dtype=np.bool)
    t = np.zeros_like(z, dtype=np.float64)

    while np.any(cond):

        t[cond] = d[cond] * ive(k, z[cond])
        S[cond] += t[cond]
        d[cond] *= x[cond]
        k += 1
      #  print(np.min(d), np.max(d))
        cond = np.abs(t/S) > eps

    return c + s * np.exp(-0.5 * (a-b)**2) * S


def estimate_sigma_grappa(data, L, grappa_kernel_W=None, cov_matrix=None, n=3):
    """Estimation of the standard deviation of noise in parallel MRI.

    data : Data to estimate the noise variance from

    theta : LxL covariance matrix of GRAPPA reconstruction weights (eq. 15-16)

    L : Number of channels in the receiver coils

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
    #return m2
    return np.sqrt(_sigma2_eff(theta, m2, L))


def _estimate_m2(data, n):

    padded = np.pad(data, ((n, n), (n, n), (n, n), (0, 0)), mode='reflect')  #add_padding_reflection(data[..., 0], n)
    #m2 = np.zeros_like(padded, dtype=np.float64)

    m2 = np.zeros_like(data[..., 0], dtype=np.float64)
    deb = time()
    for idx in ndindex(m2.shape):
        m2[idx] = np.mean(padded[idx[0]-n:idx[0]+n+1,
                                 idx[1]-n:idx[1]+n+1,
                                 idx[2]-n:idx[2]+n+1, :]**2)
        #print(idx, m2[idx])
    print("total", time() - deb)
    #m2[np.isnan(m2)] = 0
    return m2
    # pad, mais on veut juste padder 3D en realite et utiliser toutes les DWIs
    # for idx in range(data.shape[-1]):
    #     padded = np.pad(data[..., idx], (n, n), mode='reflect')

    # m2 = np.zeros((padded.shape, data.shape[-1]), dtype=np.float64)

    # for idx in range(data.shape[-1]):

    #     a = np.array(idx) + (n - 1)
    #     b = a + 2 * n + 1
    #     m2[idx] = np.mean(padded[a:b,
    #                              a:b,
    #                              a:b, :]**2)
    #     print(idx,a,b)

    # return m2[n:-n, n:-n, n:-n, :]


#def _L_eff(L, A, sigma2):
# Eq. 17 p.4


#def _sigma2_eff(phi, sigma2b, sigma2s):
# Eq. 33 p.4
#    return phi * sigma2b + (1 - phi) * sigma2s


def _sigma2_eff(theta, m2, L):
# Eq. 35 p. 6
    trace_theta = np.trace(theta)

    # Cheap mode
    round_m2 = np.round(m2).astype(np.int64)
    mode_m2 = np.bincount(round_m2[round_m2 > 0]).argmax()

    sigma2n = 0.5 * mode_m2 / trace_theta #mode(m2/trace_theta, axis=None)
    print(sigma2n)
    #sigma2n = sigma2n[0]
    tts = trace_theta * sigma2n

    return sigma2n * (tts/(m2 - tts) * (np.abs(np.sum(theta))/L)
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


def chi_to_gauss(m, eta, sigma, N, alpha=1e-7, eps=1e-7):

    m = np.array(m)
    eta = np.array(eta)
    cdf = np.zeros_like(m, dtype=np.float64)

    for idx in [np.logical_and(np.logical_and(eta/sigma < m/sigma,  np.logical_and(np.abs(eta) > eps, np.abs(m) > eps)), eta > 0),
                np.logical_and(np.logical_and(eta/sigma >= m/sigma, np.logical_and(np.abs(eta) > eps, np.abs(m) > eps)), eta > 0),
                np.abs(m) <= eps,
                np.abs(eta) <= eps]:

        if cdf[idx].size > 0:
            cdf[idx] = 1 - _marcumq_octave(eta[idx]/sigma, m[idx]/sigma, N)

    # octave code does not play well with eta < 0...
    idx = eta < 0
    print("eta < 0", np.sum(idx))
    cdf[idx] = 1 - _marcumq_matlab(eta[idx]/sigma, m[idx]/sigma, N)

   # for idx in ndindex(cdf.shape):
   #     cdf[idx] = 1 - _marcumq(eta[idx]/sigma, m[idx]/sigma, N)
    #print(cdf, 1 - _marcumq_matlab(eta[idx]/sigma, m[idx]/sigma, N), m, eta, sigma, N)
    # Find outliers and clip them to the confidence interval limits
    print("clip cdf < ", np.sum(cdf < alpha/2), " > ", np.sum(cdf > 1 - alpha/2), "out of", cdf.size, cdf.min(), cdf.max())
    np.clip(cdf, alpha/2, 1 - alpha/2, out=cdf)

    return _inv_cdf_gauss(cdf, eta, sigma)


def fixed_point_finder(m_hat, sigma, N, max_iter=100, eps=1e-4):

    m = copy(m_hat).astype(np.float32)
    delta = _beta(N) * sigma - m_hat
    out = np.zeros_like(delta, dtype=np.float32)
    t0 = np.zeros_like(delta, dtype=np.float32)
    t1 = np.zeros_like(delta, dtype=np.float32)

    for idx in [delta < 0, delta > 0]:
        ###print(idx)
        if np.all(delta[idx] > 0):
            print ("delta > 0", np.sum(delta[idx] > 0))
        elif np.all(delta[idx] < 0):
            print ("delta < 0", np.sum(delta[idx] < 0))
        else:
            print("oups")

        #if delta == 0:
        #    return 0

        #if np.all(delta[idx] != 0):
        #print(m)
        if np.all(delta[idx] > 0):
            print("shift delta")
            m[idx] = _beta(N) * sigma + delta[idx]
        #print(m)
        t0[idx] = m[idx]
        t1[idx] = _fixed_point_k(t0[idx], m[idx], sigma, N)
        ###print(t0,t1)
        ###print(_fixed_point_k(t1[idx], m[idx], sigma, N))
        #1/0
        #t0 = m[idx]
        #t1 = _fixed_point_k(t0, m[idx], sigma, N)

        n_iter = 0
        #print(t0.shape, t1.shape, idx.shape)
        ind = np.zeros_like(delta, dtype=np.bool)
        ind[idx] = np.abs(t0[idx] - t1[idx]) > eps
        #print(np.sum(np.isnan(t1[idx])), "t1")
        #print(np.sum(np.isnan(delta)), "delta")
        #while np.any(np.abs(t0 - t1) > eps):
        ###print(ind,"ind in")
        ###from copy import copy

        # Prevent looping on small non converging cases
        sum_ind0 = np.sum(ind)
        print("min, max, t0, t1", np.min(t0), np.min(t1), np.max(t0), np.max(t1))

        while np.any(ind):

            #t0 = t1
            #t1 = _fixed_point_k(t0, m[idx], sigma, N)
            #n_iter += 1
            ###print(t0, t1, ind,"cas 1", np.abs(t0[idx] - t1[idx]))
            ###print(t0.dtype, t1.dtype, ind,"dtype", np.abs(t0[idx] - t1[idx]).dtype)
            t0[ind] = t1[ind]
            t1[ind] = _fixed_point_k(t0[ind], m[ind], sigma, N)
            n_iter += 1
            ind[idx] = np.abs(t0[idx] - t1[idx]) > eps
            ###print(t0, t1, ind, "cas 2")
            sum_ind1 = np.sum(ind)
            print(np.sum(ind), "Total diff abs", np.sum(np.abs(t0[idx] - t1[idx])),
                "Max diff abs", np.max(np.abs(t0[idx] - t1[idx])))

            if n_iter > max_iter:
                print("trop d'iter :(")
                break

            if ((sum_ind0 - sum_ind1) == 0) and n_iter > 5:
                print("loop around sur ind", sum_ind0)
                break

            sum_ind0 = np.sum(ind)

        if np.all(delta[idx] > 0):
            out[idx] = -t1[idx]

        if np.all(delta[idx] < 0):
            out[idx] = t1[idx]

    # for idx in ndindex(delta.shape):
    #     if delta[idx] > 0:
    #         t1[idx] *= -1

    # return t1
    return out
    #return t1


def piesno(data, N, alpha=0.01, l=100, itermax=100, eps=1e-5):
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

    # prevent overflow in sum_m2
    data = data.astype(np.float32)
    # Get optimal quantile if available, else use the median
    q = _optimal_quantile(N)

    # Initial estimation of sigma
    denom = np.sqrt(2 * _inv_nchi_cdf(N, 1, q))
    #m = np.median(data) / denom
    m = np.percentile(data, q*100) / denom

    # More zero voxels than anything : not useable
    if m == 0:
        return 0, np.zeros_like(data[..., 0], dtype=np.bool)

    phi = np.arange(1, l + 1) * m/l
    K = data.shape[-1]
    sum_m2 = np.sum(data**2, axis=-1)
    #print(data.shape,sum_m2.dtype, np.sum(sum_m2<0), np.sum(data**2), data.min(), data.max())
    #1/0

    sigma = np.zeros_like(phi)
    mask = np.zeros(phi.shape + data.shape[:-1])
    #print(mask.shape)
    lambda_minus = _inv_nchi_cdf(N, K, alpha/2)
    lambda_plus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    pos = 0
    max_length_omega = 0

    for num, sig in enumerate(phi):

        sig_prev = 0.
        omega_size = 1
        #idx = np.zeros_like(sum_m2, dtype=np.bool)

        for n in range(itermax):

            if np.abs(sig - sig_prev) < eps:
                break

            s = sum_m2 / (2*K*sig**2)
            idx = np.logical_and(lambda_minus <= s, s <= lambda_plus)
            #print(np.sum(idx), omega_size,lambda_minus, lambda_plus, np.sum(s<0), (2*K*sig**2), np.sum(sum_m2<0), np.sum(data<0),"in")
            omega = data[idx, :]
            #print('1  ', len(omega), omega_size, omega.size, type(omega_size), type(omega.size))
            #print('2  ', np.sum(idx), 'idx')
            ##print(np.sum(idx),"idx", np.shape(idx), np.shape(omega), np.shape(data), np.abs(sig - sig_prev))
            # If no point meets the criterion, exit
            if omega.size == 0:
                omega_size = 0
                ##print("vide", num,np.abs(sig - sig_prev),"\n")
                break

            sig_prev = sig
            #sig = np.median(omega) / denom
            # Numpy percentile must range in 0 to 100, hence q*100
            sig = np.percentile(omega, q*100) / denom
            omega_size = omega.size/K
            #print(sig, n)

        # Remember the biggest omega array as giving the optimal
        # sigma amongst all initial estimates from phi
        if omega_size > max_length_omega:
            pos, max_length_omega = num, omega_size
        #print('3  ', omega_size, omega.size, type(omega_size), type(omega.size))
        sigma[num] = sig
        mask[num] = idx
        #print(np.sum(idx), omega_size, "out")
        #1/0
    return sigma[pos], mask[pos]  #, idx#[pos], idx


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
    recon = np.dot(noise_comp.T, dwis) #+ mean[-1]
    #return dwis.reshape(data.shape[:-1] + (-1,))
    #dwis += mean
    print(recon.shape, noise_comp.shape, dwis.shape, mean.shape)
    #noise_comp = np.zeros_like(U)
    #noise_comp[..., -1:] = U[..., -1:]
    #recon = np.dot(noise_comp, np.dot(noise_comp.T, dwis)) + mean

    #noise = recon.reshape(data.shape[:-1] + (-1,))
    noise = recon.reshape(data.shape[:3])

    print (noise.shape,"bla")

    return noise
    return noise_field(noise, radius)
    #s_noise = np.zeros_like(s)
    #s_noise[-1] = s[-1]
    #noise = np.dot(U * s_noise, Vt) += sub


def correction_scheme(data, N=12):
    pass


#from scilpy.denoising.utils import _im2col_3d#, _col2im_3d
def local_means(arr, radius=1):

    if arr.ndim == 3:
        arr = arr[..., None]

    out = np.zeros_like(arr)

    for i in range(arr.shape[-1]):
        temp = _im2col_3d(arr, (2*radius+1, 2*radius+1, 2*radius+1), (2*radius, 2*radius, 2*radius), 'C')

        out[..., i] = np.mean(temp).reshape(arr.shape)  #np.reshape(_col2im_3d(A, size, overlap, order)
          #  _col2im_3d(R, block_shape, end_shape, overlap, order)
        # for idx in ndindex(out.shape[:-1]):
        #     print(idx, arr.shape, out.shape, out.shape[:-1])
        #     print(idx)
        #     out[..., i] = np.mean(arr[idx[0]-radius:arr[idx[0]+radius+1]],
        #                           arr[idx[1]-radius:arr[idx[1]+radius+1]],
        #                           arr[idx[2]-radius:arr[idx[2]+radius+1]])

    return out

from scilpy.denoising.utils import im2col_nd, col2im_nd, padding
def lpca(img, sigma):

    shape = img.shape
   # img=padding(img, (3, 3, 3, shape[-1]), (2, 2, 2, 1))
    mat = im2col_nd(img, (3, 3, 3, img.shape[-1]), (2, 2, 2, 1))
    out = np.zeros_like(mat)
    thresh = 2.3 * sigma**2
    print(mat.shape, img.shape, shape)
    for i in range(mat.shape[0]):
       # print(i)
        current = np.zeros((3**3, 65), dtype=np.float32)
        for j in range(27):
            current[:, j] = mat[i,j*27:(j+1)*27]

        mean = np.mean(current, axis=0, keepdims=True)
        U, s, Vt = svd(current - mean, full_matrices=False)
        # print(current.shape, mean.shape)
       # print(np.sum(s>0),s.max())
       # s[s < thresh] = 0
        #print(np.sum(s>0), s.max())
        out[i] = np.ravel(np.dot(U * s, Vt) + mean)

        out[i] = current.ravel()
    print(out.shape, mat.shape, img.shape, shape, shape[-1])
    print((3, 3, 3, img.shape[-1]), img.shape, (2, 2, 2, 1))
    return col2im_nd(out.T, (3, 3, 3, img.shape[-1]), img.shape, (2, 2, 2, 1))


from dipy.denoise.denspeed import non_stat_noise
from scipy.ndimage import convolve


def estimate_sigma(arr):
    """Standard deviation estimation from local patches

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    Returns
    -------
    sigma : ndarray
        map of standard deviation of the noise.
    """

    if arr.ndim == 3:
        arr = arr[..., None]

    sigma = np.zeros_like(arr, dtype=np.float32)
    k = np.ones((3, 3, 3))
    temp = np.zeros_like(sigma[..., 0])
    ##k2 = np.ones((5, 5, 5), dtype=np.int16)

    for i in range(sigma.shape[-1]):
        temp = convolve(arr[..., i], k, mode='reflect', output=temp)
        sigma[..., i] = np.sqrt(non_stat_noise(arr[..., i] - temp/np.sum(k)))
        #sigma[..., i] = convolve(temp, k2)/np.sum(k2)
        #print(non_stat_noise(arr[..., i]).shape)

    return sigma

from scipy.ndimage.filters import uniform_filter, generic_filter


def local_standard_deviation(arr):
    """Standard deviation estimation from local patches

    https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    Returns
    -------
    sigma : ndarray
        map of standard deviation of the noise.
    """

    if arr.ndim == 3:
        arr = arr[..., None]

    sigma = np.zeros_like(arr, dtype=np.float32)
    size = (3, 3, 3)
    k = np.ones(size)
    temp = np.zeros_like(sigma[..., 0])

    for i in range(sigma.shape[-1]):

        convolve(arr[..., i], k, output=temp, mode='reflect')/np.sum(k)
        generic_filter(arr[..., i] - temp, np.std, size=size, mode='reflect', output=sigma[..., i])

    return sigma


# def window_stdev(arr):

#     radius = 3

#     c1 = uniform_filter(arr, radius*2, mode='reflect', origin=-radius)
#     c2 = uniform_filter(arr**2, radius*2, mode='reflect', origin=-radius)

#     return np.sqrt((c2 - c1**2)[:-radius*2+1,:-radius*2+1])
