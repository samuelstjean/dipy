#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm

from dipy.denoise.signal_transformation_framework import (_inv_cdf_gauss, _xi,
 _beta, _marcumq, _inv_nchi_cdf, piesno, chi_to_gauss, fixed_point_finder)


def test_inv_cdf_gauss():
    loc = np.random.randint(-10, 10)
    scale = np.random.rand()
    y = np.random.rand() * scale + loc
    assert_almost_equal(_inv_cdf_gauss(norm.cdf(y, loc=loc, scale=scale), loc, scale), y, decimal=10)


def test_beta():
    # Values taken from hispeed.SignalFixedPointFinder.beta
 #   assert_almost_equal(_beta(3), 2.349964007466563, decimal=10)
  #  assert_almost_equal(_beta(7), 3.675490580428171, decimal=10)
    assert_almost_equal(_beta(12), 4.848227898082543, decimal=10)


def test_xi():
    # Values taken from hispeed.SignalFixedPointFinder.xi
    assert_almost_equal(_xi(50, 2, 2), 0.9976038446303619)
    assert_almost_equal(_xi(100, 25, 12), 0.697674262651006)
    assert_almost_equal(_xi(4, 1, 12), 0.697674262651006)


def test_fixed_point_finder():
    # Values taken from hispeed.SignalFixedPointFinder.fixedPointFinder
    assert_almost_equal(fixed_point_finder(np.array([50]), 30, 12), -192.78288201533618)
    assert_almost_equal(fixed_point_finder(np.array([650]), 45, 1), 648.4366584016703)


def test_chi_to_gauss():
    # Values taken from hispeed.DistributionalMapping.nonCentralChiToGaussian
    assert_almost_equal(chi_to_gauss(np.array([470, 700]), np.array([600, 600]), 80, 12), [331.2511087335721, 586.5304199340127])
    assert_almost_equal(chi_to_gauss(np.array([700, 470]), np.array([600, 600]), 80, 1), [695.0548001366581, 463.965319619292])


def test_marcumq():
    # Values taken from octave's marcumq function
    assert_almost_equal(_marcumq(2, 1, 0),  0.730987939964090)
    assert_almost_equal(_marcumq(7, 5, 0),  0.972285213704037)
    assert_almost_equal(_marcumq(3, 7, 5),  0.00115139503866225)
    assert_almost_equal(_marcumq(0, 7, 5),  4.07324330517049e-07)
    assert_almost_equal(_marcumq(7, 0, 5),  1.)


def test_inv_nchi():
    # Values taken from hispeed.MedianPIESNO.lambdaPlus
    # and hispeed.MedianPIESNO.lambdaMinus
    N = 8
    K = 20
    alpha = 0.01

    lambdaMinus = _inv_nchi_cdf(N, K, alpha/2)
    lambdaPlus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    assert_almost_equal(lambdaMinus, 6.464855180579397)
    assert_almost_equal(lambdaPlus, 9.722849086419043)


def test_piesno():
    # Values taken from hispeed.MedianPIESNO with the test data
    # in the package computed in matlab
    sigma = piesno(test_piesno, N=8, alpha=0.01, l=1)
    assert_almost_equal(sigma, 0.010635911195599)


#test_piesno()
test_inv_nchi()
test_marcumq()
test_chi_to_gauss()
test_fixed_point_finder()
test_xi()
test_beta()
test_inv_cdf_gauss()
