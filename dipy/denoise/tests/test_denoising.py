import numpy as np

from numpy.testing import assert_array_equal

from dipy.denoise.aonlm import aonlm
from dipy.denoise.ornlm import ornlm


def test_ornlm():

    a = np.arange(1000).reshape(10,10,10)
    b = ornlm(a, 7, 3, 1)


def test_aonlm():

    a = np.arange(1000).reshape(10,10,10)
    b = aonlm(a, 7, 3, 1)
