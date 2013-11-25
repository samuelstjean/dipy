import numpy as np
from wavelet import dwt3D
from wavelet import idwt3D


def mixingsubband(fimau, fimao):
    '''
    References
    ----------
    Pierrick Coupe - pierrick.coupe@gmail.com
    Jose V. Manjon - jmanjon@fis.upv.es
    Brain Imaging Center, Montreal Neurological Institute.
    Mc Gill University
    Copyright (C) 2010 Pierrick Coupe and Jose V. Manjon
    '''
    s = fimau.shape
    p0 = 2 ** (np.ceil(np.log2(s[0])))
    p1 = 2 ** (np.ceil(np.log2(s[1])))
    p2 = 2 ** (np.ceil(np.log2(s[2])))
    pad1 = np.zeros((p0, p1, p2))
    pad2 = pad1.copy()
    pad1[:s[0], :s[1], :s[2]] = fimau
    pad2[:s[0], :s[1], :s[2]] = fimao
    af = np.array([[0, -0.01122679215254],
                   [0, 0.01122679215254],
                   [-0.08838834764832,   0.08838834764832],
                   [0.08838834764832,   0.08838834764832],
                   [0.69587998903400,  -0.69587998903400],
                   [0.69587998903400,   0.69587998903400],
                   [0.08838834764832,  -0.08838834764832],
                   [-0.08838834764832,  -0.08838834764832],
                   [0.01122679215254,                  0],
                   [0.01122679215254,                  0]])
    sf = np.array(af[::-1, :])
    w1 = dwt3D(pad1, 1, af)
    w2 = dwt3D(pad2, 1, af)

    w1[0][2] = w2[0][2]
    w1[0][4] = w2[0][4]
    w1[0][5] = w2[0][5]
    w1[0][6] = w2[0][6]

    fima = idwt3D(w1, 1, sf)
    fima = fima[:s[0], :s[1], :s[2]]

    fima[np.isnan[fima]] = 0
    fima[fima < 0] = 0
    return fima
