#! /usr/bin/env python

from __future__ import division, print_function

import nibabel as nib
import numpy as np

import os
import argparse

from dipy.denoise.signal_transformation_framework import chi_to_gauss, fixed_point_finder, piesno
from scipy.stats import mode
from scipy.ndimage.filters import gaussian_filter

DESCRIPTION = """
    Convenient script to transform noisy rician/chi-squared signals into
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
                   metavar=' ', required=False, default=4, type=int,
                   help='Number of receiver coils of the scanner for GRAPPA \
                   reconstruction. Use 1 in the case of a SENSE reconstruction. \
                   Default : 12 for the 1.5T from Sherbrooke.')

    p.add_argument('-o', action='store', dest='savename',
                   metavar='savename', required=False, default=None, type=str,
                   help='Path and prefix for the saved transformed file. \
                   The name is always appended with _stabilized.nii.gz')

    return p


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    vol = nib.load(args.input)
    data = vol.get_data().astype('float64') #[30:100,30:100, 30:40, :20]
    header = vol.get_header()
    affine = vol.get_affine()

  #  max_val = data.max()
   # min_val = data.min()
    dtype = data.dtype

    # Since negatives are allowed, convert uint to int
    if dtype.kind == 'u':
        dtype = dtype.name[1:]
    #print(data.min(), data.max())
    ##data = (data - min_val) / (max_val - min_val)
    #print(data.min(), data.max())
    if args.savename is None:
        #temp, ext = str.split(os.path.basename(args.input), '.', 1)
        if os.path.basename(args.input).endswith('.nii'):
            temp = os.path.basename(args.input)[:-4]
        elif os.path.basename(args.input).endswith('.nii.gz'):
            temp = os.path.basename(args.input)[:-7]

        filename = os.path.dirname(os.path.realpath(args.input)) + '/' + temp + "_stabilized.nii.gz"

    else:
        filename = args.savename

    N = args.N
    sigma = np.zeros(data.shape[-2], dtype=np.float64)
    mask_noise = np.zeros(data.shape[:-1], dtype=np.float64)
    eta = np.zeros_like(data, dtype=np.float64)
    data_stabilized = np.zeros_like(data, dtype=np.float64)

    from time import time
    deb = time()

    for idx in range(data.shape[-2]): #  min_slice, data.shape[-2] - min_slice): in range(25,30): #
        print("Now processing slice", idx+1, "out of", data.shape[-2])

        sigma[idx], mask_noise[..., idx] = piesno(data[..., idx, :],  N=N, l=50)
        ######m_hat = np.mean(data, axis=-1)
        ######eta[..., idx, :] = fixed_point_finder(m_hat, sigma[idx], N)

        #eta[..., idx, :] = fixed_point_finder(data[..., idx, :], sigma[idx], N)

        #print(np.sum(np.isnan(eta)), np.sum(np.isinf(eta)))
        #eta[np.isnan(eta)] = data[np.isnan(eta)]
        #print(np.sum(np.isnan(eta)), np.sum(np.isinf(eta)))
        #######data_stabilized[..., idx, :] = chi_to_gauss(data[..., idx, :], eta[..., idx, :], sigma[idx], N)
    print(sigma) #,"N=1 for piesno noise detection!")

    sigma_mode = mode(sigma, axis=None)[0]
    print(mode(sigma, axis=None))
    nib.save(nib.Nifti1Image(mask_noise.astype(np.int8), affine, header), filename + 'mask_noise.nii.gz')
   # print(sigma_mode)
    #sigma_mode = 20#N * np.max(sigma)
   # print(sigma_mode, type(sigma_mode), float(sigma_mode))

    #m_hat = np.repeat(np.mean(data, axis=-1, keepdims=True), data.shape[-1], axis=-1)
    m_hat = np.zeros_like(data, dtype=np.float64)
    for idx in range(data.shape[-1]):
        m_hat[..., idx] = gaussian_filter(data[..., idx], 0.5)

    nib.save(nib.Nifti1Image(m_hat, affine, header), filename + 'm_hat.nii.gz')
    #1/0
    ###m_hat = data
   # print(type(m_hat), type(sigma_mode), type(N))
   # m_hat = nib.load('DTIpierrickfusionx10_ps1_0_denoised.nii.gz').get_data().astype(np.float64)
    eta = fixed_point_finder(m_hat, sigma_mode, N)

    ###eta = np.repeat(eta, data.shape[-1], axis=-1)
    ###eta[..., 0] = data[..., 0]
    print(data.shape,m_hat.shape,eta.shape)
    nib.save(nib.Nifti1Image(eta.astype(dtype), affine, header), filename + '_eta')

        #eta[..., idx, :] = fixed_point_finder(data[..., idx, :], sigma[idx], N)

        #print(np.sum(np.isnan(eta)), np.sum(np.isinf(eta)))
        #eta[np.isnan(eta)] = data[np.isnan(eta)]
        #print(np.sum(np.isnan(eta)), np.sum(np.isinf(eta)))
    ##data_stabilized = chi_to_gauss(m_hat, eta, sigma_mode, N)
    data_stabilized = chi_to_gauss(data, eta, sigma_mode, N)

    print("temps total:", time() - deb)
    #print(data_stabilized.min(), data_stabilized.max())
    ##data_stabilized = data_stabilized * (max_val - min_val) + min_val
    #print(data_stabilized.min(), data_stabilized.max())
    nib.save(nib.Nifti1Image(data_stabilized.astype(dtype), affine, header), filename)


if __name__ == "__main__":
    main()
