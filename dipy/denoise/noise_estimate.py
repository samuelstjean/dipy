from __future__ import division, print_function

import numpy as np

from numba import jit, autojit
from scipy.special import gammainccinv
from scipy.ndimage.filters import convolve
from numpy.lib.stride_tricks import as_strided as ast
from scilpy.utils.angular_tools import angular_neighbors
from scipy.ndimage.interpolation import zoom
from scipy.misc import imresize

# Get optimal quantile for N if available, else use the median.
opt_quantile = {1: 0.79681213002002,
                2: 0.7306303027491917,
                4: 0.6721952960782169,
                8: 0.6254030432343569,
               16: 0.5900487123737876,
               32: 0.5641772300866416,
               64: 0.5455611840489607,
              128: 0.5322811923303339}


def _inv_nchi_cdf(N, K, alpha):
    """Inverse CDF for the noncentral chi distribution
    See [1]_ p.3 section 2.3"""
    return gammainccinv(N * K, 1 - alpha) / K


def fast_piesno(data, N=1, alpha=0.01, l=100, itermax=100, eps=1e-5, return_mask=False, init=None):

    if np.all(data == 0):
        if return_mask:
            return 0, np.zeros(data.shape[:-1], dtype=np.bool)

        return 0

    if N in opt_quantile:
        q = opt_quantile[N]
    else:
        q = 0.5

    # Initial estimation of sigma
    denom = np.sqrt(2 * _inv_nchi_cdf(N, 1, q))

    if init is None:
        m = np.percentile(data, q * 100) / denom
    else:
        m = init / denom

    phi = np.arange(1, l + 1) * m / l
    K = data.shape[-1]
    sum_m2 = np.sum(data**2, axis=-1, dtype=np.float32)

    sigma_prev = 0
    sigma = m
    prev_idx = 0
    mask = np.zeros(data.shape[:-1], dtype=np.bool)

    lambda_minus = _inv_nchi_cdf(N, K, alpha/2)
    lambda_plus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    for sigma_init in phi:

        s = sum_m2 / (2 * K * sigma_init**2)
        found_idx = np.sum(np.logical_and(lambda_minus <= s, s <= lambda_plus), dtype=np.int16)
        # print(found_idx, prev_idx, sigma_init)
        if found_idx > prev_idx:
            sigma = sigma_init
            prev_idx = found_idx
    # print(sigma * np.sqrt(K))
    for n in range(itermax):
        # print(found_idx, sigma, lambda_minus, lambda_plus, np.max(sum_m2), np.min(sum_m2), (2 * K * sigma**2), K)
        if np.abs(sigma - sigma_prev) < eps:
            break

        s = sum_m2 / (2 * K * sigma**2)
        mask[...] = np.logical_and(lambda_minus <= s, s <= lambda_plus)
        omega = data[mask, :]

        # If no point meets the criterion, exit
        if omega.size == 0:
            break

        sigma_prev = sigma

        # Numpy percentile must range in 0 to 100, hence q*100
        sigma = np.percentile(omega, q * 100) / denom
        # sigma = np.percentile(omega, 100) / denom
        # print(sigma, n, np.sum(mask))
    # print(sigma * np.sqrt(K))
    if return_mask:
        return sigma, mask

    return sigma


def piesno(data, N=1, alpha=0.01, l=100, itermax=100, eps=1e-5, return_mask=False, init=None):
    """
    Probabilistic Identification and Estimation of Noise (PIESNO)
    A routine for finding the underlying gaussian distribution standard
    deviation from magnitude signals.

    This is a re-implementation of [1]_ and the second step in the
    stabilisation framework of [2]_.

    Parameters
    -----------
    data : ndarray
        The magnitude signals to analyse. The last dimension must contain the
        same realisation of the volume, such as dMRI or fMRI data.

    N : int
        The number of phase array coils of the MRI scanner.

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

    return_mask : bool
        If True, return a mask identyfing all the pure noise voxel
        that were found.

    Returns
    --------
    sigma : float
        The estimated standard deviation of the gaussian noise.

    mask (optional): ndarray
        A boolean mask indicating the voxels identified as pure noise.

    Note
    ------
    This function assumes two things : 1. The data has a noisy, non-masked
    background and 2. The data is a repetition of the same measurements
    along the last axis, i.e. dMRI or fMRI data, not structural data like T1/T2.

    References
    ------------

    .. [1] Koay CG, Ozarslan E and Pierpaoli C.
    "Probabilistic Identification and Estimation of Noise (PIESNO):
    A self-consistent approach and its applications in MRI."
    Journal of Magnetic Resonance 2009; 199: 94-103.

    .. [2] Koay CG, Ozarslan E and Basser PJ.
    "A signal transformational framework for breaking the noise floor
    and its applications in MRI."
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """

    # Get optimal quantile for N if available, else use the median.
    opt_quantile = {1: 0.79681213002002,
                    2: 0.7306303027491917,
                    4: 0.6721952960782169,
                    8: 0.6254030432343569,
                   16: 0.5900487123737876,
                   32: 0.5641772300866416,
                   64: 0.5455611840489607,
                  128: 0.5322811923303339}

    if N in opt_quantile:
        q = opt_quantile[N]
    else:
        q = 0.5

    # Initial estimation of sigma
    denom = np.sqrt(2 * _inv_nchi_cdf(N, 1, q))

    if init is None:
        m = np.percentile(data, q * 100) / denom
    else:
        m = init / denom
    phi = np.arange(1, l + 1) * m / l
    K = data.shape[-1]
    sum_m2 = np.sum(data**2, axis=-1, dtype=np.float32)

    sigma = np.zeros_like(phi)
    mask = np.zeros(phi.shape + data.shape[:-1])

    lambda_minus = _inv_nchi_cdf(N, K, alpha/2)
    lambda_plus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    pos = 0
    max_length_omega = 0

    for num, sig in enumerate(phi):

        sig_prev = 0
        omega_size = 1
        idx = np.zeros(sum_m2.shape, dtype=np.bool)

        for n in range(itermax):

            if np.abs(sig - sig_prev) < eps:
                break

            s = sum_m2 / (2 * K * sig**2)
            idx = np.logical_and(lambda_minus <= s, s <= lambda_plus)
            omega = data[idx, :]

            # If no point meets the criterion, exit
            if omega.size == 0:
                omega_size = 0
                break

            sig_prev = sig
            # Numpy percentile must range in 0 to 100, hence q*100
            sig = np.percentile(omega, q * 100) / denom
            omega_size = omega.size / K

        # Remember the biggest omega array as giving the optimal
        # sigma amongst all initial estimates from phi
        if omega_size > max_length_omega:
            pos, max_length_omega, best_mask = num, omega_size, idx

        sigma[num] = sig
        mask[num] = idx

    if return_mask:
        return sigma[pos], mask[pos]

    return sigma[pos]


def estimate_sigma(arr, disable_background_masking=False):
    """Standard deviation estimation from local patches

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    disable_background_masking : bool, default False
        If True, uses all voxels for the estimation, otherwise, only non-zeros voxels are used.
        Useful if the background is masked by the scanner.

    Returns
    -------
    sigma : ndarray
        standard deviation of the noise, one estimation per volume.
    """
    k = np.zeros((3, 3, 3), dtype=np.int8)

    k[0, 1, 1] = 1
    k[2, 1, 1] = 1
    k[1, 0, 1] = 1
    k[1, 2, 1] = 1
    k[1, 1, 0] = 1
    k[1, 1, 2] = 1

    if arr.ndim == 3:
        sigma = np.zeros(1, dtype=np.float32)
        arr = arr[..., None]
    elif arr.ndim == 4:
        sigma = np.zeros(arr.shape[-1], dtype=np.float32)
    else:
        raise ValueError("Array shape is not supported!", arr.shape)

    if disable_background_masking:
        mask = arr[..., 0].astype(np.bool)
    else:
        mask = np.ones_like(arr[..., 0], dtype=np.bool)

    conv_out = np.zeros(arr[..., 0].shape, dtype=np.float64)
    for i in range(sigma.size):
        convolve(arr[..., i], k, output=conv_out)
        mean_block = np.sqrt(6/7) * (arr[..., i] - 1/6 * conv_out)
        sigma[i] = np.sqrt(np.mean(mean_block[mask]**2))

    return sigma


def local_piesno(data, bvals, bvecs, N=1, block_size=6, size=5, return_mask=True):

    if N in opt_quantile:
        q = opt_quantile[N]
    else:
        q = 0.5

    b0_loc = np.argmax(bvals == 0)
    num_b0s = np.sum(bvals == 0)
    sym_bvecs = np.vstack((np.delete(bvecs, b0_loc, axis=0), np.delete(-bvecs, b0_loc, axis=0)))
    neighbors = (angular_neighbors(sym_bvecs, block_size) % (data.shape[-1] - num_b0s))[:data.shape[-1] - num_b0s]
    # print(neighbors.shape, data.shape)
    # 1/0
    init = np.percentile(data, q*100)

    dwi = np.delete(data, b0_loc, axis=-1)
    noise_dwi = np.zeros(dwi.shape, dtype=np.float32)
    s_out = np.zeros((dwi.shape[0]//size, dwi.shape[1]//size, dwi.shape[2]//size, dwi.shape[-1]), dtype=np.float32)
    m_out = np.zeros(dwi.shape[:-1], dtype=np.bool)

    for n in range(dwi.shape[-1]):
        cur = neighbors[n].tolist()
        cur.append(n)

        mean_dwi = np.mean(dwi[..., cur], axis=-1)
        # mean2_dwi = np.sum(np.mean(dwi[..., cur], axis=-1)**2)
        noise_dwi[..., n] = (dwi[..., n] - mean_dwi) #- mean2_dwi)
        # noise_dwi[..., n] = dwi[..., n] - mean_dwi
        # print(noise_dwi.shape, (size,size,size, last_dim_size), mean_dwi.shape, noise_dwi.shape, cur, dwi[..., cur].shape)
    # print(np.min(noise_dwi), np.max(noise_dwi), np.min(data), np.max(data), np.median(data), np.median(noise_dwi))
    # print(np.percentile(data, 80), np.percentile(noise_dwi, 80))
    # print(mean_dwi.shape, mean2_dwi.shape, noise_dwi.shape, b0_loc, num_b0s)
    # return noise_dwi, noise_dwi

    # for n in [noise_dwi.shape[-1]]:#range(noise_dwi.shape[-1]):
    # cur = range(noise_dwi.shape[-1]) #neighbors[n].tolist()
    # cur.append(n)

    reshaped_maps = sliding_window(noise_dwi, (size, size, size, noise_dwi.shape[-1]))

    # mean2 = np.zeros(reshaped_maps.shape[0])
    # for i in cur:
    #     mean2 += np.mean(sliding_window(noise_dwi[..., i], (size, size, size)).reshape(-1, size**3), axis=-1)**2

    # reshaped_maps -= np.sqrt(mean2[:, None, None, None, None])
    # reshaped_maps = np.abs(reshaped_maps)
    # print(mean2.shape, reshaped_maps.shape)
    sigma = np.zeros(reshaped_maps.shape[0], dtype=np.float32)
    mask = np.zeros((reshaped_maps.shape[0], size**3), dtype=np.bool)

    for i in range(reshaped_maps.shape[0]):
        # print(i)
        cur_map = reshaped_maps[i].reshape(size**3, 1, -1)
        sigma[i], m = fast_piesno(cur_map, N=N, return_mask=True, init=init, alpha=0.1)
        mask[i] = np.squeeze(m)
        sigma[i] = np.std(cur_map)

    s_out[..., 0] = sigma.reshape(noise_dwi.shape[0]//size, noise_dwi.shape[1]//size, noise_dwi.shape[2]//size)

    # for n in range(dwi.shape[-1]):

    #     cur = neighbors[n].tolist()
    #     cur.append(n)
    #     reshaped_maps = sliding_window(dwi[..., cur], (size,size,size, block_size))
    #     sigma = np.zeros(reshaped_maps.shape[0], dtype=np.float32)
    #     mask = np.zeros((reshaped_maps.shape[0], size**3), dtype=np.bool)

    #     for i in range(reshaped_maps.shape[0]):
    #         cur_map = reshaped_maps[i].reshape(size**3, 1, -1)
    #         sigma[i], m = fast_piesno(cur_map, N=N, return_mask=True, init=init)
    #         mask[i] = np.squeeze(m)

    #     s_out[..., n] = sigma.reshape(dwi.shape[0]//size, dwi.shape[1]//size, dwi.shape[2]//size)

        # n = 0
        # print(mask.shape, noise_dwi.shape[:-1], m_out[:-size, :-size, :-size].shape)
    n = 0
    for i in np.ndindex(s_out.shape[:-1]):
        i = np.array(i) * size
        j = i + size
        # print(i,j,n)
        m_out[i[0]:j[0], i[1]:j[1], i[2]:j[2]] = mask[n].reshape(size, size, size)
        n += 1

    # interpolated1 = zoom(np.squeeze(s_out[..., 0]), np.array(data[..., 0].shape) / np.array(np.squeeze(s_out[..., 0]).shape), order=1)
    interpolated = np.zeros_like(data[..., 0], dtype=np.float32)
    # x, y, z = np.array(data[..., 0].shape) - np.array(s_out[..., 0].shape) * size
    x, y, z = np.array(s_out[..., 0].shape) * size
    interpolated[:x, :y, :z] = zoom(np.squeeze(s_out[..., 0]), size, order=1)

    if return_mask:
        print('return mask')
        return s_out, m_out

    return s_out, interpolated


@jit(nogil=True, cache=True)
def inner_piesno(data, m, q, l, denom, lambda_plus, lambda_minus):

    itermax = 100
    eps = 1e-5

    phi = np.arange(1, l + 1) * m / l
    K = data.shape[0]

    pos = 0
    max_length_omega = 0
    sigma = np.zeros_like(phi)
    mask = np.zeros((len(phi), K))

    for num, sig in enumerate(phi):

        sig_prev = 0
        omega_size = 1
        idx = np.zeros(data.shape)

        for n in range(itermax):

            if np.abs(sig - sig_prev) < eps:
                break

            s = data / (2 * K * sig**2)
            idx = np.logical_and(lambda_minus <= s, s <= lambda_plus)
            omega = data[idx]

            # If no point meets the criterion, exit
            if omega.size == 0:
                break

            sig_prev = sig
            # Numpy percentile must range in 0 to 100, hence q*100
            sig = np.percentile(omega, q*100) / denom
            # sig = percentile(omega, q) / denom

        # Remember the biggest omega array as giving the optimal
        # sigma amongst all initial estimates from phi
        if omega_size > max_length_omega:
            pos, max_length_omega = num, omega_size

        sigma[num] = sig
        mask[num] = idx

    return sigma[pos], mask[pos]

def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

import math
import functools


def percentile(N, percent, key=lambda x:x):
    """
    Find the percentile of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile of the values
    """
    if not N:
        return None
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0+d1


def jensen_shannon_divergence(a, b):
    """Compute Jensen-Shannon Divergence
    Parameters
    ----------
    a : array-like
        possibly unnormalized distribution.
    b : array-like
        possibly unnormalized distribution. Must be of same shape as ``a``.
    Returns
    -------
    j : float
    See Also
    --------
    jsd_matrix : function
        Computes all pair-wise distances for a set of measurements
    entropy : function
        Computes entropy and K-L divergence
    """
    a = np.asanyarray(a, dtype=float)
    b = np.asanyarray(b, dtype=float)
    a = a/a.sum(axis=0)
    b = b/b.sum(axis=0)
    m = (a + b)
    m /= 2.
    m = np.where(m, m, 1.)
    #return 0.5 * np.sum(a * np.log2(a/m) + b * np.log2(b/m), axis=0)
    return 0.5*np.sum(xlogy(a, a/m) + xlogy(b, b/m), axis=0)


def beta(N):
    return np.sqrt(np.pi / 2) * factorial2(2*N - 1) / (2**(N-1) * factorial(N-1))


def local_snr(data, bvals, bvecs, size=5, block_size=5):

    b0_loc = np.argmax(bvals == 0)
    num_b0s = np.sum(bvals == 0)
    sym_bvecs = np.vstack((np.delete(bvecs, b0_loc, axis=0), np.delete(-bvecs, b0_loc, axis=0)))
    neighbors = (angular_neighbors(sym_bvecs, block_size - num_b0s) % (data.shape[-1] - num_b0s))[:data.shape[-1] - num_b0s]

    dwi = np.delete(data, b0_loc, axis=-1)
    noise_dwi = np.zeros_like(dwi, dtype=np.float32)

    for n in range(dwi.shape[-1]):
        cur = neighbors[n].tolist()
        cur.append(n)

        mean_dwi = np.mean(dwi[..., cur], axis=-1)
        noise_dwi[..., n] = np.abs(dwi[..., n] - mean_dwi)

    SNR = np.zeros_like(dwi, dtype=np.float32)

    for n in range(SNR.shape[-1]):
        reshaped_maps = sliding_window(noise_dwi[..., n], (size,size,size))
        reshaped_maps.shape = (reshaped_maps.shape[0], -1)
        local_snr = np.mean(reshaped_maps, axis=-1) /  np.std(reshaped_maps, axis=-1)

        out = local_snr.reshape(dwi.shape[0]//size, dwi.shape[1]//size, dwi.shape[2]//size)
        x, y, z = np.array(out.shape) * size
        SNR[:x, :y, :z, n] = zoom(out, size, order=1)

    return SNR

def _local_SNR(N):
    return beta(N) / np.sqrt(2*N - beta(N)**2)
