#from __future__ import division, print_function

import numpy as np

cimport numpy as cnp
cimport cython

from scipy.special import iv

def test():
    pass

from dipy.denoise.denspeed import add_padding_reflection, remove_padding

def _chi(SNR):
    return 2 + SNR**2 - np.pi/8 * np.exp(-SNR**2/2) * ((2 + SNR**2) * iv(0, SNR**2/4) + SNR**2 * iv(1, SNR**2/4))**2



@cython.wraparound(False)
@cython.boundscheck(False)
def noise_field(double[:, :, ::1] arr, patch_radius):

    arr = add_padding_reflection(arr, patch_radius)

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double[:, :, ::1] out = np.zeros_like(arr)
        cnp.npy_intp P = patch_radius
        double std = 0
        double mean = 0

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    # move the block
    for i in range(P, I - P):
        for j in range(P, J - P):
            for k in range(P, K - P):

                std = np.std(arr[i-P:i+P, j-P:j+P, k-P:k+P])
                mean = np.mean(arr[i-P:i+P, j-P:j+P, k-P:k+P])
                out[i, j, k] = std**2 / _chi(mean/std)

    return remove_padding(out, patch_radius)
