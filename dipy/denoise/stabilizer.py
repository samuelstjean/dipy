#! /usr/bin/env python

from __future__ import division, print_function

import nibabel as nib
import numpy as np

import os
import argparse

from multiprocessing import Pool
from itertools import repeat

from dipy.denoise.signal_transformation_framework import piesno, local_standard_deviation #, _fixed_point_finder, _chi_to_gauss, fixed_point_finder
from dipy.denoise.denspeed import _chi_to_gauss, fixed_point_finder, corrected_sigma

from scipy.stats import mode
from scipy.ndimage.filters import gaussian_filter, convolve
# from dipy.denoise.nlmeans import nlmeans
from dipy.core.ndindex import ndindex
# from skimage.restoration import denoise_bilateral


DESCRIPTION = """
    Convenient script to transform noisy rician/non central chi signals into
    gaussian distributed signals.

    Reference:
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('input', action='store', metavar=' ',
                   help='Path of the image file to stabilize.')

    p.add_argument('-N', action='store', dest='N',
                   metavar=' ', required=True, type=int,
                   help='Number of receiver coils of the scanner for GRAPPA \
                   reconstruction. Use 1 in the case of a SENSE reconstruction.')

    p.add_argument('-o', action='store', dest='savename',
                   metavar='savename', required=False, default=None, type=str,
                   help='Path and prefix for the saved transformed file. \
                   The name is always appended with _stabilized.nii.gz')

    return p


def helper(arglist):

    data, m_hat, sigma, N = arglist
    out = np.zeros(data.shape, dtype=np.float32)

    for idx in ndindex(data.shape):

        if sigma[idx] > 0:
            sigma_corr = corrected_sigma(m_hat[idx], sigma[idx], N)
            eta = fixed_point_finder(m_hat[idx], sigma_corr, N)
            out[idx] = _chi_to_gauss(data[idx], eta, sigma_corr, N)
        else:
            out[idx] = 0

    return out


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    vol = nib.load(args.input)
    data = vol.get_data()
    header = vol.get_header()
    affine = vol.get_affine()

    dtype = data.dtype
    #data = data[:, 20:30, ...]
    # Since negatives are allowed, convert uint to int
    if dtype.kind == 'u':
        dtype = dtype.name[1:]

    if args.savename is None:
        if os.path.basename(args.input).endswith('.nii'):
            temp = os.path.basename(args.input)[:-4]
        elif os.path.basename(args.input).endswith('.nii.gz'):
            temp = os.path.basename(args.input)[:-7]

        filename = os.path.split(os.path.abspath(args.input))[0] + '/' + temp
        print("savename is", filename)

    else:
        filename = args.savename

    N = args.N
    data = data.astype(np.float32)
    sigma = np.zeros(data.shape[-2], dtype=np.float32)
    mask_noise = np.zeros(data.shape[:-1], dtype=np.bool)
    #eta = np.zeros_like(data, dtype=np.float32)
    #data_stabilized = np.zeros_like(data, dtype=np.int16)

    from time import time
    deb = time()



    # for idx in range(data.shape[-2]):
    #     print("Now processing slice", idx+1, "out of", data.shape[-2])
    #     sigma[idx], mask_noise[..., idx] = piesno(data[..., idx, :],  N)

    # print(sigma)
    # print(np.percentile(sigma, 10.),  np.percentile(sigma, 90.))

    # #sigma_mode = np.load(filename + "_sigma.npy")

    # sigma_mode, num = mode(sigma, axis=None)
    # # sigma_mode=200.#25.62295723
    # print("mode of sigma is", sigma_mode, "with nb", num, "median is", np.median(sigma))
    # np.save(filename + "_sigma.npy", sigma_mode)
    # nib.save(nib.Nifti1Image(mask_noise.astype(np.int8), affine, header), filename + '_mask_noise.nii.gz')



    m_hat = np.zeros_like(data, dtype=np.float32)
    k = np.ones((3, 3, 3))
    for idx in range(data.shape[-1]):
        #m_hat[..., idx] = gaussian_filter(data[..., idx], 0.5)
        m_hat[..., idx] = convolve(data[..., idx], k) / np.sum(k)
        # cur_max = np.max(data[..., idx])

    # m_hat = nlmeans(data, sigma_mode, rician=False)
    ### m_hat = data
    # m_hat *= mask_noise[..., None]

    # sigma_mat = np.ones_like(m_hat, dtype=np.float32) * sigma_mode
    sigma_mat = local_standard_deviation(data)
    # nib.save(nib.Nifti1Image(sigma_mat, np.eye(4)), 'sigmat.nii.gz')
    # sigma_mat = np.median(sigma_mat, axis=-1)
    # nib.save(nib.Nifti1Image(sigma_mat, np.eye(4)), 'sigmatmed.nii.gz')

    # sigma_mat = np.ones_like(data) * sigma_mat[..., None]
    # np.save(filename + "_sigma.npy", sigma_mat)
    nib.save(nib.Nifti1Image(sigma_mat, affine, header), filename + '_sigma.nii.gz')
   # m_hat = nib.load('/home/local/USHERBROOKE/stjs2902/Bureau/phantomas_mic/b1000/dwis.nii.gz').get_data()
    nib.save(nib.Nifti1Image(m_hat, affine, header), filename + '_m_hat.nii.gz')
    #sigma_mode=515.
    # m_hat = nib.load('/home/local/USHERBROOKE/stjs2902/Bureau/phantomas_mic/b1000/dwis.nii.gz').get_data()
    n_cores = 8
    n = data.shape[-2]
    nbr_chunks = n_cores
    chunk_size = int(np.ceil(n / nbr_chunks))
    #data = data[..., 0]
    #m_hat = m_hat[..., 0]
    #chunk_size=1

    pool = Pool(processes=n_cores)
    arglist=[(data_vox, m_hat_vox, sigma_vox, N_vox) for data_vox, m_hat_vox, sigma_vox, N_vox in zip(data, m_hat, repeat(sigma_mat), repeat(N))]
    data_stabilized = pool.map(helper, arglist, chunksize=chunk_size)
    #print(arglist[0], 'bla')
    #out =  pool.map(helper, arglist)
    #data_stabilized = np.asarray(data_stabilized).reshape(data.shape)
    #print(data_stabilized.shape)

    pool.close()
    pool.join()
    data_stabilized = np.asarray(data_stabilized).reshape(data.shape)
    print(data_stabilized.shape)


    #eta = fixed_point_finder(m_hat, sigma_mode, N)
    #print(data.shape, m_hat.shape, eta.shape)
    #nib.save(nib.Nifti1Image(eta.astype(dtype), affine, header), filename + '_eta.nii.gz')
    #data_stabilized = chi_to_gauss(data, eta, sigma_mode, N)

    print("temps total:", time() - deb)
    nib.save(nib.Nifti1Image(data_stabilized.astype(dtype), affine, header), filename + "_stabilized.nii.gz")

    # print("Detected noise std was :", sigma_mode)


if __name__ == "__main__":
    main()
