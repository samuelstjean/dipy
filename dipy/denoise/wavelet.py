from __future__ import division, print_function

import numpy as np
from dipy.denoise.filters import upfir, firdn


def permutationInverse(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def sfb3D_A(lo, hi, sf, d):
    lpf = sf[:, 0]
    hpf = sf[:, 1]
    # permute dimensions of lo and hi so that dimension d is first.
    p = [(i + d) % 3 for i in range(3)]
    lo = lo.transpose(p)
    hi = hi.transpose(p)

    (N1, N2, N3) = lo.shape
    N = 2 * N1
    L = sf.shape[0]
    y = np.zeros((N + L - 2, N2, N3))
    for k in range(N3):
        y[:, :, k] = (np.array(upfir(lo[:, :, k], lpf)) +
                      np.array(upfir(hi[:, :, k], hpf)))

    y[:(L - 2), :, :] = y[:(L - 2), :, :] + y[N:(N + L - 2), :, :]
    y = y[:N, :, :]
    y = cshift3D(y, 1 - L / 2, 0)

    # permute dimensions of y (inverse permutation)
    q = reversed(p)
    #q = permutationInverse(p)
    y = y.transpose(q)
    return y


def sfb3D(lo, hi, sf1, sf2=None, sf3=None):

    if sf2 is None:
        sf2 = sf1

    if sf3 is None:
        sf3 = sf1

    LLL = lo
    LLH = hi[0]
    LHL = hi[1]
    LHH = hi[2]
    HLL = hi[3]
    HLH = hi[4]
    HHL = hi[5]
    HHH = hi[6]

    # filter along dimension 2
    LL = sfb3D_A(LLL, LLH, sf3, 2)
    LH = sfb3D_A(LHL, LHH, sf3, 2)
    HL = sfb3D_A(HLL, HLH, sf3, 2)
    HH = sfb3D_A(HHL, HHH, sf3, 2)

    # filter along dimension 1
    L = sfb3D_A(LL, LH, sf2, 1)
    H = sfb3D_A(HL, HH, sf2, 1)

    # filter along dimension 0
    y = sfb3D_A(L, H, sf1, 0)

    return y


def afb3D_A(x, af, d):
    lpf = af[:, 0]
    hpf = af[:, 1]

    # permute dimensions of x so that dimension d is first.
    p = [(i + d) % 3 for i in range(3)]
    x = x.transpose(p)

    # filter along dimension 0
    (N1, N2, N3) = x.shape
    L = af.shape[0] // 2
    x = cshift3D(x, -L, 0)
    n1Half = N1 // 2
    lo = np.zeros((L + n1Half, N2, N3))
    hi = np.zeros((L + n1Half, N2, N3))
    for k in range(N3):
        lo[:, :, k] = firdn(x[:, :, k], lpf)

    lo[:L] = lo[:L] + lo[n1Half:n1Half+L, :, :]
    lo = lo[:n1Half, :, :]

    for k in range(N3):
        hi[:, :, k] = firdn(x[:, :, k], hpf)

    hi[:L] = hi[:L]+hi[n1Half:n1Half+L, :, :]
    hi = hi[:n1Half, :, :]

    # permute dimensions of x (inverse permutation)
    q = reversed(p)
    #q = permutationInverse(p)
    lo = lo.transpose(q)
    hi = hi.transpose(q)
    return lo, hi


def afb3D(x, af1, af2=None, af3=None):

    if af2 is None:
        af2 = af1

    if af3 is None:
        af3 = af1

    # filter along dimension 0
    L, H = afb3D_A(x, af1, 0)

    # filter along dimension 1
    LL, LH = afb3D_A(L, af2, 1)
    HL, HH = afb3D_A(H, af2, 1)

    # filter along dimension 2
    LLL, LLH = afb3D_A(LL, af3, 2)
    LHL, LHH = afb3D_A(LH, af3, 2)
    HLL, HLH = afb3D_A(HL, af3, 2)
    HHL, HHH = afb3D_A(HH, af3, 2)

    return LLL, [LLH, LHL, LHH, HLL, HLH, HHL, HHH]


def idwt3D(w, J, sf):
    y = w[J]
    for k in range(J)[::-1]:
        y = sfb3D(y, w[k], sf, sf, sf)

    return y


def dwt3D(x, J, af):
    w = [None] * (J + 1)
    for k in range(J):
        x, w[k] = afb3D(x, af, af, af)

    w[J] = x
    return w


def cshift3D(x, m, d):
    s = x.shape
    idx = (np.array(range(s[d])) + (s[d] - m % s[d])) % s[d]
    if d == 0:
        return x[idx, :, :]
    elif d == 1:
        return x[:, idx, :]
    else:
        return x[:, :, idx]
