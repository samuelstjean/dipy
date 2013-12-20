#! /usr/bin/env python

from __future__ import division, print_function

import nibabel as nib
import numpy as np

import os
import argparse

from dipy.denoise.signal_transformation_framework import chi_to_gauss, fixed_point_finder, piesno
#from dipy.core.ndindex import ndindex

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

    #p.add_argument('bvals', action='store', metavar=' ',
    #               help='Path of the bvals file, in FSL format.')

    p.add_argument('-N', action='store', dest='N',
                   metavar=' ', required=False, default=12, type=int,
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
    data = vol.get_data()#[30:100,30:100, 30:40, :20]
    header = vol.get_header()
    affine = vol.get_affine()

    max_val = data.max()
    min_val = data.min()
    dtype = data.dtype
    data = (data - min_val) / (max_val - min_val)

    #o_dtype = data.dtype
    #o_shape = data.shape
    #data = data.astype('float64')
    #data_stabilized = np.zeros_like(data)
    #bvals, _ = read_bvals_bvecs(args.bvals, None)

    if args.savename is None:
        temp, ext = str.split(os.path.basename(args.input), '.', 1)
        filename = os.path.dirname(os.path.realpath(args.input)) + '/' + temp + "_stabilized.nii.gz"

    else:
        filename = args.savename

    N = args.N

    # Initialize Java VM
    #hispeed.initVM()

    # Estimated noise standard deviation
    #sigma = np.zeros(data.shape[-1])
    print("Now running PIESNO...")
    #tima=time()

    #data = data[..., 2:]

    #min_slice = data.shape[-2]//4
    sigma = np.zeros(data.shape[-2])
    #eta = np.zeros(data.shape[-2])

    eta = np.zeros_like(data)


    #N=1
    #sigma = []
    #eta = []
    #eta = np.mean(data, axis=tuple(range(data.ndim-1)))
    #print(tuple(range(data.ndim)))
    data_stabilized = np.zeros_like(data, dtype=np.float64)
    #mean_slice = np.mean(data.swapaxes(-1, -2), axis=tuple(range(data.ndim-1)))
    #print(mean_slice.shape)

    for idx in range(data.shape[-2]): #  min_slice, data.shape[-2] - min_slice):
   #     print("Now processing slice", idx+1, "out of", data.shape[-2])
        sigma[idx] = piesno(data[..., idx, :],  N)
        #eta[idx] = fixed_point_finder(mean_slice[idx], sigma[idx], N)

        #signal_intensity = fixed_point_finder(np.mean(data[..., idx, :]), sigma[idx], N)
        #print(sigma[idx], eta[idx], np.mean(data[..., idx, :]), np.median(data[..., idx, :]), np.mean(data), np.median(data))
        #print(sigma[idx])


    #for idx in range(data.shape[-2]):
        #print(idx)
        eta[..., idx, :] = fixed_point_finder(data[..., idx, :], sigma[idx], N)
        #data_stabilized[idx] = chi_to_gauss(data[idx], eta[idx], sigma[idx[-2]], N)
        #print(idx)
        #print (idx, eta[30, 30, idx, 0], data[30, 30, idx, 0], sigma[idx])
        #if idx[0] > cur:
        #    cur = idx[0]
        #    print(cur, eta[30, 30, idx[-2], 0], data[30, 30, idx[-2], 0],
        #          data_stabilized[30, 30, idx[-2], 0], sigma[idx[-2]])


    # cur = 0
    # for idx in ndindex(data.shape):
    #     #print(idx)
    #     eta[idx] = fixed_point_finder(data[idx], sigma[idx[-2]], N)
    #     #data_stabilized[idx] = chi_to_gauss(data[idx], eta[idx], sigma[idx[-2]], N)
    #     #print(idx)

    #     if idx[0] > cur:
    #         cur = idx[0]
    #         print(cur, eta[30, 30, idx[-2], 0], data[30, 30, idx[-2], 0],
    #               data_stabilized[30, 30, idx[-2], 0], sigma[idx[-2]])

        #print(eta[idx])

    #1/0

    #sigma = piesno(data,  N)
        #print(mean_slice[idx], sigma[idx], N)
    #eta = fixed_point_finder(np.mean(mean_slice), sigma, N)

    #print(sigma)
    #print(eta)
    #1/0
    #for idx in range(data.shape[-2]):
        print(idx)
        #print(chi_to_gauss(data[..., idx, :], eta[idx], sigma[idx], N))
        data_stabilized[..., idx, :] = chi_to_gauss(data[..., idx, :], eta[..., idx, :], sigma[idx], N)

    #print(np.median(sigma), np.mean(sigma))
    #sigma = np.median(sigma)
    #signal_intensity = fixed_point_finder(np.mean(data), sigma, N)
    #print(signal_intensity, np.mean(data))

        #sigma[idx] = 20
        #print(eta.shape)
        #print(eta[idx], eta.shape, data.shape)
        #signal_intensity = fixed_point_finder(eta[idx], sigma[idx], N)
        #signal_intensity = 120
        #print(sigma[idx], signal_intensity)
        #data_stabilized[..., idx] = chi_to_gauss(data[..., idx], signal_intensity, sigma[idx], N)

    data_stabilized = data_stabilized * (max_val - min_val) + min_val

    nib.save(nib.Nifti1Image(data_stabilized.astype(dtype), affine, header), filename)


if __name__ == "__main__":
    main()
