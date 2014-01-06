#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm

from dipy.denoise.signal_transformation_framework import (_inv_cdf_gauss,
     chi_to_gauss, _xi, fixed_point_finder, _beta, _marcumq)

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
assert_almost_equal(chi_to_gauss(np.array([470, 700, 0]), 600, 80, 12), [331.2511087335721, 586.5304199340127, 321.53948765230064])
assert_almost_equal(chi_to_gauss(np.array([700, 0, 470]), 600, 80, 1), [695.0548001366581, 321.53948765230064, 463.965319619292])

# Values taken from octave
assert_almost_equal(_marcumq(7, 3, 5),  0.999999658508735)
assert_almost_equal(_marcumq(3, 7, 5),  0.00115139503866225)
assert_almost_equal(_marcumq(0, 7, 5),  4.07324330517049e-07)
assert_almost_equal(_marcumq(7, 0, 5),  1.)