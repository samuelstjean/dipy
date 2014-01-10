#! /usr/bin/env python

from __future__ import division, print_function

import nibabel as nib
import numpy as np

import os
import argparse

from dipy.denoise.signal_transformation_framework import chi_to_gauss, fixed_point_finder, piesno

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
    data = vol.get_data().astype('float64') #[30:100,30:100, 30:40, :20]
    header = vol.get_header()
    affine = vol.get_affine()

    max_val = data.max()
    min_val = data.min()
    dtype = data.dtype

    # Since negatives are allowed, convert uint to int
    if dtype.kind == 'u':
        dtype = dtype.name[1:]

    ##data = (data - min_val) / (max_val - min_val)

    if args.savename is None:
        temp, ext = str.split(os.path.basename(args.input), '.', 1)
        filename = os.path.dirname(os.path.realpath(args.input)) + '/' + temp + "_stabilized.nii.gz"

    else:
        filename = args.savename

    N = args.N
    sigma = np.zeros(data.shape[-2], dtype=np.float64)
    eta = np.zeros_like(data, dtype=np.float64)
    data_stabilized = np.zeros_like(data, dtype=np.float64)

    from time import time
    deb = time()

    for idx in range(data.shape[-2]): #  min_slice, data.shape[-2] - min_slice): in range(25,30): #
        print("Now processing slice", idx+1, "out of", data.shape[-2])

        sigma[idx] = piesno(data[..., idx, :],  N)
        eta[..., idx, :] = fixed_point_finder(data[..., idx, :], sigma[idx], N)

        #print(np.sum(np.isnan(eta)), np.sum(np.isinf(eta)))
        #eta[np.isnan(eta)] = data[np.isnan(eta)]
        #print(np.sum(np.isnan(eta)), np.sum(np.isinf(eta)))
        data_stabilized[..., idx, :] = chi_to_gauss(data[..., idx, :], eta[..., idx, :], sigma[idx], N)

    print("temps total:", time() - deb)

    ##data_stabilized = data_stabilized * (max_val - min_val) + min_val
    nib.save(nib.Nifti1Image(data_stabilized.astype(dtype), affine, header), filename)


if __name__ == "__main__":
    main()
