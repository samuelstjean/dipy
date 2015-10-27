from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange

#from dipy.denoise.signal_transformation_framework import _inv_cdf_gauss
# from scilpy.denoising.hyp1f1 import hyp1f1_pas_vec as hyp1f1
from scipy.special import erfinv

from libc.math cimport sqrt, exp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


def nlmeans_3d(arr, mask=None, sigma=None, patch_radius=1,
               block_radius=5, rician=True):
    """ Non-local means for denoising 3D images

    Parameters
    ----------
    arr : 3D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.
    """

    if arr.ndim != 3:
        raise ValueError('data needs to be a 3D ndarray', arr.shape)

    if mask is None:

        mask = np.ones(arr.shape, dtype='f8')

    else:
        mask = np.ascontiguousarray(mask, dtype='f8')

    if mask.ndim != 3:
        raise ValueError('mask needs to be a 3D ndarray', mask.shape)

    if sigma.ndim != 3:
        raise ValueError('sigma needs to be a 3D ndarray', sigma.shape)

    arr = np.ascontiguousarray(arr, dtype='f8')
    arr = add_padding_reflection(arr, block_radius)
    mask = add_padding_reflection(mask.astype('f8'), block_radius)
    sigma = np.ascontiguousarray(sigma, dtype='f8')
    sigma = add_padding_reflection(sigma.astype('f8'), block_radius)
    arrnlm = _nlmeans_3d(arr, mask, sigma, patch_radius, block_radius, rician)


    return remove_padding(arrnlm, block_radius)


def nlmeans_4d(arr, mask=None, sigma=None, patch_radius=1,
               block_radius=5, rician=True):
    """ Non-local means for denoising 3D images

    Parameters
    ----------
    arr : 3D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.

    """
    if arr.ndim != 4:
        raise ValueError('arr needs to be a 3D ndarray')

    if mask is None:
        mask = np.ones_like(arr[..., 0], dtype='f8', order='C')
    else:
        mask = np.ascontiguousarray(mask, dtype='f8')

    if mask.ndim != 3:
        raise ValueError('arr needs to be a 3D ndarray')

    arr = np.ascontiguousarray(arr, dtype='f8')
    arr = add_padding_reflection4D(arr, block_radius)
    mask = add_padding_reflection(mask.astype('f8'), block_radius)
    arrnlm = _nlmeans_4d(arr, mask, sigma, patch_radius, block_radius, rician)


    return remove_padding(arrnlm, block_radius)


@cython.wraparound(False)
@cython.boundscheck(False)
def _nlmeans_3d(double [:, :, ::1] arr, double [:, :, ::1] mask,
                double [:, :, ::1] sigma, patch_radius=1, block_radius=5,
                rician=True):
    """ This algorithm denoises the value of every voxel (i, j, k) by
    calculating a weight between a moving 3D patch and a static 3D patch
    centered at (i, j, k). The moving patch can only move inside a
    3D block.
    """

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double [:, :, ::1] out = np.zeros_like(arr)
        double summ = 0
        cnp.npy_intp P = patch_radius
        cnp.npy_intp B = block_radius

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    # move the block
    with nogil, parallel():
        for i in prange(B, I - B):
            for j in range(B, J - B):
                for k in range(B, K - B):

                    if mask[i, j, k] == 0:
                        continue

                    out[i, j, k] = process_block(arr, i, j, k, B, P, sigma)

    new = np.asarray(out)

    if rician:
        new -= 2 * np.asarray(sigma)**2
        new[new < 0] = 0

    return np.sqrt(new)


@cython.wraparound(False)
@cython.boundscheck(False)
def _nlmeans_4d(double [:, :, :, ::1] arr, double [:, :, ::1] mask,
                sigma, patch_radius=1, block_radius=5,
                rician=True):
    """ This algorithm denoises the value of every voxel (i, j ,k) by
    calculating a weight between a moving 3D patch and a static 3D patch
    centered at (i, j, k). The moving patch can only move inside a
    3D block.
    """

    cdef:
        cnp.npy_intp i, j, k, l, I, J, K, L
        double [:, :, :, ::1] out = np.zeros_like(arr)
        double summ = 0
        double sigm = 0
        cnp.npy_intp P = patch_radius
        cnp.npy_intp B = block_radius

    sigm = sigma

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]
    L = arr.shape[3]

    #move the block
    with nogil, parallel():
        for i in prange(B, I - B):
            for j in range(B , J - B):
                for k in range(B, K - B):

                    if mask[i, j, k] == 0:
                        continue

                    for l in range(L):
                        out[i, j, k, l] = process_block4D(arr, i, j, k, l, B, P, sigm)

    new = np.asarray(out)

    if rician:
        new -= 2 * sigm ** 2
    new[new < 0] = 0

    return np.sqrt(new)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double process_block(double [:, :, ::1] arr,
                          cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k,
                          cnp.npy_intp B, cnp.npy_intp P, double [:, :, ::1] sigma) nogil:
    """ Process the block with center at (i, j, k)

    Parameters
    ----------
    arr : 3D array
        C contiguous array of doubles
    i, j, k : int
        center of block
    B : int
        block radius
    P : int
        patch radius
    sigma : 3D array
        local noise standard deviation

    Returns
    -------
    new_value : double
    """

    cdef:
        cnp.npy_intp m, n, o, M, N, O, a, b, c, cnt, step
        double patch_vol_size
        double summ, d, w, sumw, sum_out, x, sigm
        double * W
        double * cache
        double * sigma_block
        double denom
        cnp.npy_intp BS = B * 2 + 1


    cnt = 0
    sumw = 0
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)

    W = <double *> malloc(BS * BS * BS * sizeof(double))
    cache = <double *> malloc(BS * BS * BS * sizeof(double))
    sigma_block = <double *> malloc(BS * BS * BS * sizeof(double))

    # (i, j, k) coordinates are the center of the static patch
    # copy block in cache
    copy_block_3d(cache, BS, BS, BS, arr, i - B, j - B, k - B)
    copy_block_3d(sigma_block, BS, BS, BS, sigma, i - B, j - B, k - B)

    # calculate weights between the central patch and the moving patch in block
    # (m, n, o) coordinates are the center of the moving patch
    # (a, b, c) run inside both patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):

                summ = 0
                sigm = 0

                # calculate square distance
                for a in range(-P, P + 1):
                    for b in range(-P, P + 1):
                        for c in range(-P, P + 1):

                            # this line takes most of the time! mem access
                            d = cache[(B + a) * BS * BS + (B + b) * BS + (B + c)] - cache[(m + a) * BS * BS + (n + b) * BS + (o + c)]
                            summ += d * d
                            sigm += sigma_block[(m + a) * BS * BS + (n + b) * BS + (o + c)]

                denom = sqrt(2) * (sigm / patch_vol_size)**2
                w = exp(-(summ / patch_vol_size) / denom)
                sumw += w
                W[cnt] = w
                cnt += 1

    cnt = 0
    sum_out = 0

    # calculate normalized weights and sums of the weights with the positions
    # of the patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):

                if sumw > 0:
                    w = W[cnt] / sumw
                else:
                    w = 0

                x = cache[m * BS * BS + n * BS + o]
                sum_out += w * x * x
                cnt += 1

    free(W)
    free(cache)
    free(sigma_block)

    return sum_out


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double process_block4D(double [:, :, :, ::1] arr,
                          cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k, cnp.npy_intp l,
                          cnp.npy_intp B, cnp.npy_intp P, double sigma) nogil:
    """ Process the block with center at (i, j, k)

    Parameters
    ----------
    arr : 3D array
        C contiguous array of doubles
    i, j, k : int
        center of block
    B : int
        block radius
    P : int
        patch radius
    sigma : double

    Returns
    -------
    new_value : double
    """

    cdef:
        cnp.npy_intp m, n, o, p, M, N, O, Q, a, b, c, e, cnt, step, last_dim
        double patch_vol_size
        double summ, d, w, sumw, sum_out, x
        double * W
        double * cache
        double denom
        cnp.npy_intp BS = B * 2 + 1

    cnt = 0
    sumw = 0
    last_dim = arr.shape[3]
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1) * (last_dim)
    denom = sigma * sigma

    W = <double *> malloc(BS * BS * BS * last_dim * sizeof(double))
    cache = <double *> malloc(BS * BS * BS * last_dim * sizeof(double))

    # (i, j, k) coordinates are the center of the static patch
    # copy block in cache
    copy_block_4d(cache, BS, BS, BS, last_dim, arr, i - B, j - B, k - B, 0)
  #  with gil:
   #     print("truc2.5")
   # with gil:
        #print("truc")
    # calculate weights between the central patch and the moving patch in block
    # (m, n, o) coordinates are the center of the moving patch
    # (a, b, c) run inside both patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):
                #for p in range(last_dim):
                    #p = 0
                    summ = 0
                  #  with gil:
                   #     print(m,n,o)

                    # calculate square distance
                    for a in range(- P, P + 1):
                        for b in range(- P, P + 1):
                            for c in range(- P, P + 1):
                                for e in range(last_dim):

                                    # this line takes most of the time! mem access
                                    d = cache[(B + a) * BS * BS + (B + b) * BS + (B + c) + e] - cache[(m + a) * BS * BS + (n + b) * BS + (o + c) + e]
                                    summ += d * d

                    w = exp(-(summ / patch_vol_size) / denom)
                    sumw += w
                    W[cnt] = w
                    cnt += 1


    cnt = 0
    sum_out = 0
  #  with gil:
   #     print("out")
    # calculate normalized weights and sums of the weights with the positions
    # of the patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):
                for p in range(last_dim):

                    if sumw > 0:
                        w = W[cnt] / sumw
                    else:
                        w = 0

                    x = cache[m * BS * BS + n * BS + o + p]

                    sum_out += w * x * x

                    cnt += 1

    free(W)
    free(cache)

    return sum_out


def add_padding_reflection(double [:, :, ::1] arr, padding):
    cdef:
        double [:, :, ::1] final
        cnp.npy_intp i, j, k
        cnp.npy_intp B = padding
        cnp.npy_intp [::1] indices_i = correspond_indices(arr.shape[0], padding)
        cnp.npy_intp [::1] indices_j = correspond_indices(arr.shape[1], padding)
        cnp.npy_intp [::1] indices_k = correspond_indices(arr.shape[2], padding)

    final = np.zeros(np.array((arr.shape[0], arr.shape[1], arr.shape[2])) + 2*padding)

    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            for k in range(final.shape[2]):
                final[i, j, k] = arr[indices_i[i], indices_j[j], indices_k[k]]

    return final


def add_padding_reflection4D(double [:, :, :, ::1] arr, padding):
    cdef:
        double [:, :, :, ::1] final
        cnp.npy_intp i, j, k, l
        cnp.npy_intp B = padding
        cnp.npy_intp [::1] indices_i = correspond_indices(arr.shape[0], padding)
        cnp.npy_intp [::1] indices_j = correspond_indices(arr.shape[1], padding)
        cnp.npy_intp [::1] indices_k = correspond_indices(arr.shape[2], padding)
        #cnp.npy_intp [::1] indices_l = correspond_indices(arr.shape[3], padding)

    final = np.zeros(np.hstack((np.array((arr.shape[0], arr.shape[1], arr.shape[2])) + 2*padding, arr.shape[3])))

    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            for k in range(final.shape[2]):
                    final[i, j, k, :] = arr[indices_i[i], indices_j[j], indices_k[k], :]

    return final


def correspond_indices(dim_size, padding):
    return np.ascontiguousarray(np.hstack((np.arange(1, padding + 1)[::-1],
                                np.arange(dim_size),
                                np.arange(dim_size - padding - 1, dim_size - 1)[::-1])),
                                dtype=np.intp)


def remove_padding(arr, padding):
    shape = arr.shape
    return arr[padding:shape[0] - padding,
               padding:shape[1] - padding,
               padding:shape[2] - padding]


# def remove_padding4D(arr, padding):
#     shape = arr.shape
#     return arr[padding:shape[0] - padding,
#                padding:shape[1] - padding,
#                padding:shape[2] - padding,
#                padding:shape[3] - padding]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef cnp.npy_intp copy_block_3d(double * dest,
                                cnp.npy_intp I,
                                cnp.npy_intp J,
                                cnp.npy_intp K,
                                double [:, :, ::1] source,
                                cnp.npy_intp min_i,
                                cnp.npy_intp min_j,
                                cnp.npy_intp min_k) nogil:

    cdef cnp.npy_intp i, j

    for i in range(I):
        for j in range(J):
            memcpy(&dest[i * J * K  + j * K], &source[i + min_i, j + min_j, min_k], K * sizeof(double))

    return 1



@cython.wraparound(False)
@cython.boundscheck(False)
cdef cnp.npy_intp copy_block_4d(double * dest,
                                 cnp.npy_intp I,
                                 cnp.npy_intp J,
                                 cnp.npy_intp K,
                                 cnp.npy_intp L,
                                 double [:, :, :, ::1] source,
                                 cnp.npy_intp min_i,
                                 cnp.npy_intp min_j,
                                 cnp.npy_intp min_k,
                                 cnp.npy_intp min_l) nogil:

    cdef cnp.npy_intp i, j, k

    for i in range(I):
        for j in range(J):
            for k in range(K):
                memcpy(&dest[i * J * K * L + j * K * L + k * L], &source[i + min_i, j + min_j, k + min_k, min_l], L * sizeof(double))

    return 1


def non_stat_noise(arr, mask=None,  patch_radius=1, block_radius=5):
    """ Non-local means for denoising 3D images

    Parameters
    ----------
    arr : 3D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.

    """
    if arr.ndim != 3:
        raise ValueError('arr needs to be a 3D ndarray')

    if mask is None:
        mask = np.ones_like(arr, dtype='f8', order='C')
    else:
        mask = np.ascontiguousarray(mask, dtype='f8')

    if mask.ndim != 3:
        raise ValueError('arr needs to be a 3D ndarray')

    arr = np.ascontiguousarray(arr, dtype='f8')
    arr = add_padding_reflection(arr, block_radius)
    mask = add_padding_reflection(mask.astype('f8'), block_radius)
    arrnlm = _non_stat_noise(arr, mask, patch_radius, block_radius)

    return remove_padding(arrnlm, block_radius)


@cython.wraparound(False)
@cython.boundscheck(False)
def _non_stat_noise(double [:, :, ::1] arr, double [:, :, ::1] mask,
                    patch_radius=1, block_radius=5):
    """ This algorithm denoises the value of every voxel (i, j, k) by
    calculating a weight between a moving 3D patch and a static 3D patch
    centered at (i, j, k). The moving patch can only move inside a
    3D block.
    """

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double [:, :, ::1] out = np.zeros_like(arr)
        cnp.npy_intp P = patch_radius
        cnp.npy_intp B = block_radius

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    #move the block
    with nogil, parallel():
        for i in prange(B, I - B):
            for j in range(B, J - B):
                for k in range(B, K - B):

                    if mask[i, j, k] == 0:
                        continue

                    out[i, j, k] = block_variance(arr, i, j, k, B, P)

    return np.asarray(out)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double block_variance(double [:, :, ::1] arr,
                           cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k,
                           cnp.npy_intp B, cnp.npy_intp P) nogil:
    """ Process the block with center at (i, j, k)

    Parameters
    ----------
    arr : 3D array
        C contiguous array of doubles
    i, j, k : int
        center of block
    B : int
        block radius
    P : int
        patch radius
    sigma : double

    Returns
    -------
    new_value : double
    """

    cdef:
        cnp.npy_intp m, n, o, a, b, c
        double patch_vol_size
        double d, sumd, mind
        double * cache
        cnp.npy_intp BS = B * 2 + 1

    min_d = 1e15
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)
    cache = <double *> malloc(BS * BS * BS * sizeof(double))

    # (i, j, k) coordinates are the center of the static patch
    # copy block in cache
    copy_block_3d(cache, BS, BS, BS, arr, i - B, j - B, k - B)

    # Compute the mean of each block
    # (m, n, o) coordinates are the center of the moving patch
    # (a, b, c) run inside both patches

    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):

                sumd = 0

                for a in range(-P, P + 1):
                    for b in range(-P, P + 1):
                        for c in range(-P, P + 1):

                            # this line takes most of the time! mem access
                            d = cache[(B + a) * BS * BS + (B + b) * BS + (B + c)] - cache[(m + a) * BS * BS + (n + b) * BS + (o + c)]
                            sumd += d**2

                if (sumd < min_d) and (sumd > 0):
                    min_d = sumd

    free(cache)

    return min_d


# def _inv_cdf_gauss(y, eta, sigma):
#     """Helper function for _chi_to_gauss. Returns the gaussian distributed value
#     associated to a given probability. See p. 4 of [1] eq. 13.

#     y : float
#         Probability of observing the desired value in the normal distribution N(eta, sigma**2)
#     eta :
#         Mean of the normal distribution N(eta, sigma**2)
#     sigma : float
#         Standard deviation of the normal distribution N(eta, sigma**2)

#     return :
#         Value associated to probability y given a normal distribution N(eta, sigma**2)
#     """
#     sqrt2 = 1.4142135623730951
#     return eta + sigma * sqrt2 * erfinv(2*y - 1)


# def _chi_to_gauss(m, eta, sigma, N, alpha=1e-7):
#     """Maps the noisy signal intensity from a Rician/Non central chi distribution
#     to it's gaussian counterpart. See p. 4 of [1] eq. 12.

#     m : float
#         The noisy, Rician/Non central chi distributed value
#     eta : float
#         The underlying signal intensity estimated value
#     sigma : float
#         The gaussian noise estimated standard deviation
#     N : int
#         Number of coils of the acquision (N=1 for Rician noise)
#     alpha : float
#         Confidence interval for the cumulative distribution function.
#         clips the cdf to alpha/2 <= cdf <= 1-alpha/2

#     return
#         float : The noisy gaussian distributed signal intensity
#     """

#     cdf = 1 - _marcumq_cython(eta/sigma, m/sigma, N)
#     cdf = np.clip(cdf, alpha/2, 1 - alpha/2)
#     return _inv_cdf_gauss(cdf, eta, sigma)


# @cython.cdivision(True)
# cdef _marcumq_cython(double a, double b, int M, double eps=1e-8, int max_iter=10000):
#     """Computes the generalized Marcum Q function of order M.
#     http://en.wikipedia.org/wiki/Marcum_Q-function

#     a : float, eta/sigma
#     b : float, m/sigma
#     M : int, order of the function (Number of coils, N=1 for Rician noise)

#     return : float
#         Value of the function, always between 0 and 1 since it's a pdf.
#     """
#     cdef:
#         double a2, b2, d, h, f, f_err, errbnd, S, factorial_M = 1.
#         int i, j, k

#     a2 = 0.5 * a**2
#     b2 = 0.5 * b**2
#     d = exp(-a2)
#     h = exp(-a2)

#     for i in range(1, M+1):
#         factorial_M *= i

#     f = (b2**M) * exp(-b2) / factorial_M
#     f_err = exp(-b2)
#     errbnd = 1. - f_err

#     k = 1
#     S = f * h
#     j = (errbnd > 4*eps) #and ((1 - S) > 8*eps)

#     while j or k <= M:

#         d *= a2/k
#         h += d
#         f *= b2 / (k + M)
#         S += (f * h)

#         f_err *= b2 / k
#         errbnd -= f_err
#         j = (errbnd > 4*eps) #and ((1 - S) > 8*eps)
#         k += 1

#         if (k > max_iter):
#             break

#     return 1 - S


# def fixed_point_finder(m_hat, sigma, N, max_iter=100, eps=1e-4):
#     """Fixed point formula for finding eta. Table 1 p. 11 of [1].
#     This simply wraps the cython function _fixed_point_finder

#     m_hat : float
#         initial value for the estimation of eta.
#     sigma : float
#         Gaussian standard deviation of the noise.
#     N : int
#         Number of coils of the acquision (N=1 for Rician noise).
#     max_iter : int, default=100
#         maximum number of iterations before breaking from the loop.
#     eps : float, default = 1e-4
#         Criterion for reaching convergence between two subsequent estimates of eta.

#     return
#     t1 : float
#         Estimation of the underlying signal value
#     """

#     return _fixed_point_finder(m_hat, sigma, N, max_iter, eps)


# @cython.cdivision(True)
# cdef _fixed_point_finder(double m_hat, double sigma, int N, int max_iter=100, double eps=1e-4):
#     """Fixed point formula for finding eta. Table 1 p. 11 of [1]

#     m_hat : float
#         initial value for the estimation of eta
#     sigma : float
#         Gaussian standard deviation fo tthe noise
#     N : int
#         Number of coils of the acquision (N=1 for Rician noise)
#     max_iter : int, default=100
#         maximum number of iterations before breaking from the loop
#     eps : float, default = 1e-4
#         Criterion for reaching convergence between two subsequent estimates

#     return
#     t1 : float
#         Estimation of the underlying signal value
#     """

#     cdef:
#         double delta, m, t0, t1
#         int cond, n_iter

#     delta = _beta(N) * sigma - m_hat

#     if delta == 0:
#         return 0
#     elif delta > 0:
#         m = _beta(N) * sigma + delta
#     else:
#         m = m_hat

#     t0 = m
#     t1 = _fixed_point_k(t0, m, sigma, N)
#     cond = True
#     n_iter = 0

#     while cond:

#         t0 = t1
#         t1 = _fixed_point_k(t0, m, sigma, N)
#         n_iter += 1
#         cond = abs(t1 - t0) > eps

#         if n_iter > max_iter:
#             break

#     if delta > 0:
#         return -t1

#     return t1


# @cython.cdivision(False)
# cdef _beta(int N):
#     """Helper function for _xi, see p. 3 [1] just after eq. 8."""
#     #return np.sqrt(np.pi/2) * (factorial2(2*N-1)/(2**(N-1) * factorial(N-1)))

#     cdef double facN = 1., fac2N = 1., sqrtpi2 = 1.2533141373155001

#     # factorial(N-1)
#     for i in range(1, N):
#         facN *= i

#     # factorial2(2*N-1)
#     for i in range(1, 2*N, 2):
#         fac2N *= i

#     return sqrtpi2 * (fac2N / (2**(N-1) * facN))

#     # if N == 1:
#     #     return 1.25331413732
#     # elif N == 2:
#     #     return 1.87997120597
#     # elif N == 4:
#     #     return 2.74162467538
#     # elif N == 6:
#     #     return 3.39276053578
#     # elif N == 8:
#     #     return 3.93802562189
#     # elif N == 12:
#     #     return 4.84822789808
#     # elif N == 16:
#     #     return 5.61283938922
#     # elif N == 20:
#     #     return 6.28515420794
#     # elif N == 24:
#     #     return 6.89221524065
#     # elif N == 36:
#     #     return 8.45587062694
#     # elif N == 48:
#     #     return 9.77247710766
#     # elif N == 64:
#     #     return 11.2916332015
#     # else:
#     #     raise NotImplementedError("Number of coils " + N + " not supported! \
#     #         Add it using the formula in dipy.denoise.denspeed._beta")


# @cython.cdivision(True)
# cdef _fixed_point_g(double eta, double m, double sigma, int N):
#     """Helper function for _fixed_point_k, see p. 3 [1] eq. 11."""
#     return sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)


# @cython.cdivision(True)
# cdef _fixed_point_k(eta, m, sigma, N):
#     """Helper function for _fixed_point_, see p. 11 [1] eq. D2."""

#     cdef:
#         double fpg, num, denom
#         double eta2sigma = -eta**2/(2*sigma**2)

#     fpg = _fixed_point_g(eta, m, sigma, N)
#     num = fpg * (fpg - eta)

#     denom = eta * (1 - ((_beta(N)**2)/(2*N)) *
#                    hyp1f1(-0.5, N, eta2sigma) *
#                    hyp1f1(0.5, N+1, eta2sigma)) - fpg

#     return eta - num / denom


# def corrected_sigma(eta, sigma, N):
#     """Compute the local corrected standard deviation for the adaptive nonlocal means
#         according to the correction factor xi.

#     eta : float
#         Signal intensity
#     sigma : float
#         Noise magnitude standard deviation
#     N : int
#         Number of coils of the acquisition (N=1 for Rician noise)

#     return :
#         Corrected sigma value, where sigma_gaussian = sigma / sqrt(xi)
#     """

#     return sigma / sqrt(_xi(eta, sigma, N))


# @cython.cdivision(True)
# cdef _xi(double eta, double sigma, int N):
#     """Standard deviation scaling factor formula, see p. 3 of [1], eq. 10.

#     eta : float
#         Signal intensity
#     sigma : float
#         Noise magnitude standard deviation
#     N : int
#         Number of coils of the acquisition (N=1 for Rician noise)

#     return :
#         the correction factor xi, where
#         sigma_gaussian = sigma / xi
#     """

#     return 2*N + eta**2/sigma**2 - (_beta(N) * hyp1f1(-0.5, N, -eta**2/(2*sigma**2)))**2


# # Test for cython functions

# def _test_marcumq_cython(a, b, M, eps=1e-7, max_iter=10000):
#     return _marcumq_cython(a, b, M, eps, max_iter)


# def _test_beta(N):
#     return _beta(N)


# def _test_fixed_point_g(eta, m, sigma, N):
#     return _fixed_point_g(eta, m, sigma, N)


# def _test_fixed_point_k(eta, m, sigma, N):
#     return _fixed_point_k(eta, m, sigma, N)


# def _test_xi(eta, sigma, N):
#     return _xi(eta, sigma, N)
