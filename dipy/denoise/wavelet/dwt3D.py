from __future__ import division, print_function

from dipy.denoise.wavelet.afb3D import afb3D


def dwt3D(x, J, af):
    w = [None] * (J + 1)
    for k in range(J):
        x, w[k] = afb3D(x, af, af, af)

    w[J] = x
    return w
