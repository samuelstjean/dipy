#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm

from dipy.denoise.signal_transformation_framework import (_inv_cdf_gauss,
     chi_to_gauss, _xi, fixed_point_finder, _beta)

loc = np.random.randint(-10, 10)
scale = np.random.rand()
y = np.random.rand() * scale + loc
assert_almost_equal(_inv_cdf_gauss(norm.cdf(y, loc=loc, scale=scale), loc, scale), y, decimal=10)

# Values taken from hispeed.SignalFixedPointFinder.beta
assert_almost_equal(_beta(3), 2.349964007466563, decimal=10)
assert_almost_equal(_beta(7), 3.675490580428171, decimal=10)
assert_almost_equal(_beta(12), 4.848227898082543, decimal=10)

# Values taken from hispeed.SignalFixedPointFinder.xi
assert_almost_equal(_xi(50, 2, 2), 0.9976038446303619)
assert_almost_equal(_xi(100, 25, 12), 0.697674262651006)
assert_almost_equal(_xi(4, 1, 12), 0.697674262651006)

# Values taken from hispeed.SignalFixedPointFinder.fixedPointFinder
assert_almost_equal(fixed_point_finder(50, 30, 12), -192.78288201533618, decimal=10)
assert_almost_equal(fixed_point_finder(650,45,1), 648.4366584016703, decimal=10)

# Values taken from hispeed.DistributionalMapping.nonCentralChiToGaussian
assert_almost_equal(chi_to_gauss(470, 600, 80, 12), 331.2511087335721)
assert_almost_equal(chi_to_gauss(700, 600, 80, 1), 695.0548001366581)
