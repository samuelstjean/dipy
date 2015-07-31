#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm

from scilpy.denoising.stabilizer import (_test_marcumq_cython,
        _test_beta, _test_fixed_point_g, _test_fixed_point_k,
        _test_xi, _test_multifactorial, _test_inv_cdf_gauss,
        fixed_point_finder, chi_to_gauss)


def test_inv_cdf_gauss():
    loc = np.random.randint(-10, 10)
    scale = np.random.rand()
    y = np.random.rand() * scale + loc
    assert_almost_equal(_test_inv_cdf_gauss(norm.cdf(y, loc=loc, scale=scale), loc, scale), y, decimal=10)


def test_beta():
    # Values taken from hispeed.SignalFixedPointFinder.beta
 #   assert_almost_equal(_beta(3), 2.349964007466563, decimal=10)
  #  assert_almost_equal(_beta(7), 3.675490580428171, decimal=10)
    assert_almost_equal(_test_beta(12), 4.848227898082543, decimal=10)


def test_xi():
    # Values taken from hispeed.SignalFixedPointFinder.xi
    assert_almost_equal(_test_xi(50, 2, 2), 0.9976038446303619)
    assert_almost_equal(_test_xi(100, 25, 12), 0.697674262651006)
    assert_almost_equal(_test_xi(4, 1, 12), 0.697674262651006)


def test_fixed_point_finder():
    # Values taken from hispeed.SignalFixedPointFinder.fixedPointFinder
    assert_almost_equal(fixed_point_finder(np.array([50]), 30, 12), -192.78288201533618)
    assert_almost_equal(fixed_point_finder(np.array([650]), 45, 1), 648.4366584016703)


def test_chi_to_gauss():
    # Values taken from hispeed.DistributionalMapping.nonCentralChiToGaussian
    assert_almost_equal(chi_to_gauss(470, 600, 80, 12), 331.2511087335721, decimal=3)
    assert_almost_equal(chi_to_gauss(700, 600, 80, 12), 586.5304199340127, decimal=3)
    assert_almost_equal(chi_to_gauss(700, 600, 80, 1), 695.0548001366581, decimal=3)
    assert_almost_equal(chi_to_gauss(470, 600, 80, 1), 463.965319619292, decimal=3)


def test_marcumq():
    # Values taken from octave's marcumq function
    assert_almost_equal(_test_marcumq_cython(2, 1, 0),  0.730987939964090, decimal=6)
    assert_almost_equal(_test_marcumq_cython(7, 5, 0),  0.972285213704037, decimal=6)
    assert_almost_equal(_test_marcumq_cython(3, 7, 5),  0.00115139503866225, decimal=6)
    assert_almost_equal(_test_marcumq_cython(0, 7, 5),  4.07324330517049e-07, decimal=6)
    assert_almost_equal(_test_marcumq_cython(7, 0, 5),  1., decimal=6)


# def test_inv_nchi():
#     # Values taken from hispeed.MedianPIESNO.lambdaPlus
#     # and hispeed.MedianPIESNO.lambdaMinus
#     N = 8
#     K = 20
#     alpha = 0.01

#     lambdaMinus = _test_inv_nchi_cdf(N, K, alpha/2)
#     lambdaPlus = _test_inv_nchi_cdf(N, K, 1 - alpha/2)

#     assert_almost_equal(lambdaMinus, 6.464855180579397)
#     assert_almost_equal(lambdaPlus, 9.722849086419043)


def test_piesno():
    pass
    # Values taken from hispeed.MedianPIESNO with the test data
    # in the package computed in matlab
    # sigma = piesno(test_piesno, N=8, alpha=0.01, l=1)
    # assert_almost_equal(sigma, 0.010635911195599)


#test_piesno()
# test_inv_nchi()
test_marcumq()
test_chi_to_gauss()
test_fixed_point_finder()
test_xi()
test_beta()
test_inv_cdf_gauss()
