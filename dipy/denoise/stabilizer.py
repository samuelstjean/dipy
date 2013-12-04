#! /usr/bin/env python

from __future__ import division, print_function

import nibabel as nib
import numpy as np

import os
import argparse
#import hispeed

from dipy.io.gradients import read_bvals_bvecs
from dipy.denoise import piesno
from dipy.denoise.signal_transformation_framework import chi_to_gauss, fixed_point_finder

from time import time

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

    p.add_argument('bvals', action='store', metavar=' ',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('-N', action='store', dest='N',
                   metavar=' ', required=False, default=12, type=int,
                   help='Number of recever coils of the scanner. \
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
    data = vol.get_data()
    header = vol.get_header()
    affine = vol.get_affine()

    #o_dtype = data.dtype
    #o_shape = data.shape
    #data = data.astype('float64')
    #data_stabilized = np.zeros_like(data)
    #bvals, _ = read_bvals_bvecs(args.bvals, None)

    if args.savename is None:
        temp, ext = str.split(os.path.basename(args.input), '.', 1)
        filename = os.path.dirname(os.path.realpath(args.input)) + '/' + temp

    else:
        filename = args.savename

    N = args.N

    # Initialize Java VM
    #hispeed.initVM()

    # Estimated noise standard deviation
    #sigma = np.zeros(data.shape[-1])
    print("Now running PIESNO...")
    #tima=time()

    sigma = np.zeros(data.shape[-1])
    eta = np.mean(data, axis=-1)
    data_stabilized = np.zeros_like(data)

    for idx in range(data.shape[-1]):

        sigma[idx] = piesno(data[..., idx])
        signal_intensity = fixed_point_finder(eta[idx], sigma[idx], N)
        print(sigma[idx], signal_intensity)
        data_stabilized[..., idx] = chi_to_gauss(data[..., idx], signal_intensity, sigma[idx], N)


    nib.save(nib.Nifti1Image(data_stabilized, affine, header), filename)


    #sigma=14.438470881
    #sigma=[]



    #for i in range(data.shape[3]):
    #    sigma += [piesno(data[:,:,i,:], l=50,N=1)]
    #    print("Finished, found sigma was", sigma[i], "Runtime was", time()-tima)
    #print(sigma)
    #print("median", np.median(sigma), "mean", np.mean(sigma))


    #sigma = np.median(sigma)
    #1/0
    # Constant set in [1]
    #alpha = 0.01

    # Constant for stabilizer
    #alpha = 0.0005
    #degree = 4
    #nknots = 5

    # data_flat = []
    # i, j = 0, 0
    # for idx in range(np.prod(data.shape[2:])):
    #     i = j
    #     j = data.shape[idx] + i
    #     #data_flat[i:j] = data.ravel()[i:j]
    #     data_flat += data.ravel()[i:j].tolist()
    # print (data.shape, len(data_flat), time()-timea)
    #data = hispeed.JArray(data[:,8,9,5])

    #data = hispeed.JArray(data.reshape(o_shape[0], o_shape[1], -1))

    # arr = hispeed.JArray('object')(3)
    # arr[0] = hispeed.JArray('float')([10.])
    # arr[1] = hispeed.JArray('float')([10.])
    # arr[2] = hispeed.JArray('float')([1.])
    # sigma = hispeed.OptimalPIESNO(arr, alpha, args.N)#.getEstimatedGaussianNoiseSD()
    # 1/0


    #timea=time()
    #import itertools
    #for _ in range(data.ndim):
    #    data = list(itertools.chain.from_iterable(data))
    #print (len(data))

    #data_flat = []
    #data_flat = data.ravel()
    #bvals_vol = np.zeros_like(data, dtype='float64')

    #for idx in range(data.shape[-1]):
    #    bvals_vol[..., idx] = np.ones_like(data[..., idx]) * bvals[idx]

    #bvals = bvals_vol.ravel().astype('float64')
    #bvals = np.ones_like(data) * bvals
    # i, j = 0, 0
    # for idx in range(np.prod(data.shape)):
    #     i = j
    #     j = data.shape[idx] + i
    #     #data_flat[i:j] = data.ravel()[i:j]
    #     data_flat += data.ravel()[i:j].tolist()
    # print (data.shape, data.size, len(data_flat), time()-timea)

    #noiseFloorBreaker = hispeed.OneDimensionalSignalTransformationalFramework(bvals.tolist(), data_flat, args.N, sigma)

    # bvalues = np.array([ 0.0 , 35.35353535353536 , 70.70707070707071 , 106.06060606060606 ,
    #                    141.41414141414143 , 176.76767676767676 , 212.12121212121212 ,
    #                    247.47474747474746 , 282.82828282828285 , 318.1818181818182 , 353.5353535353535 ,
    #                     388.88888888888886 , 424.24242424242425 , 459.59595959595964 , 494.9494949494949 ,
    #                     530.3030303030304 , 565.6565656565657 , 601.010101010101 , 636.3636363636364 ,
    #                     671.7171717171717 , 707.070707070707 , 742.4242424242425 , 777.7777777777777 ,
    #                     813.1313131313132 , 848.4848484848485 , 883.838383838384 , 919.1919191919193 ,
    #                     954.5454545454545 , 989.8989898989898 , 1025.2525252525252 , 1060.6060606060607 ,
    #                     1095.959595959596 , 1131.3131313131314 , 1166.6666666666665 , 1202.020202020202 ,
    #                      1237.3737373737374 , 1272.7272727272727 , 1308.080808080808 , 1343.4343434343434 ,
    #                      1378.7878787878788 , 1414.141414141414 , 1449.4949494949494 , 1484.848484848485 ,
    #                      1520.2020202020203 , 1555.5555555555554 , 1590.9090909090908 , 1626.2626262626263 ,
    #                       1661.6161616161617 , 1696.969696969697 , 1732.3232323232323 , 1767.676767676768 ,
    #                        1803.030303030303 , 1838.3838383838386 , 1873.7373737373737 , 1909.090909090909 ,
    #                         1944.4444444444446 , 1979.7979797979797 , 2015.1515151515152 , 2050.5050505050503 ,
    #                          2085.8585858585857 , 2121.2121212121215 , 2156.5656565656564 , 2191.919191919192 ,
    #                           2227.272727272727 , 2262.626262626263 , 2297.979797979798 , 2333.333333333333 ,
    #                           2368.686868686869 , 2404.040404040404 , 2439.3939393939395 , 2474.747474747475 ,
    #                            2510.10101010101 , 2545.4545454545455 , 2580.808080808081 , 2616.161616161616 ,
    #                             2651.5151515151515 , 2686.868686868687 , 2722.222222222222 , 2757.5757575757575 ,
    #                              2792.929292929293 , 2828.282828282828 , 2863.636363636364 , 2898.989898989899 ,
    #                              2934.343434343434 , 2969.69696969697 , 3005.050505050505 , 3040.4040404040406 ,
    #                              3075.7575757575755 , 3111.111111111111 , 3146.4646464646466 , 3181.8181818181815 ,
    #                              3217.1717171717173 , 3252.5252525252527 , 3287.878787878788 , 3323.2323232323233 ,
    #                               3358.5858585858587 , 3393.939393939394 , 3429.2929292929293 , 3464.6464646464647 ,
    #                                3500.0])

    # noisySI = np.array([ 974.4680312435205 ,920.5810792869341 ,819.6650525922458 ,802.5190824927682 ,
    #                    727.7646357302121 ,673.6333509410033 ,637.1774633934145 ,602.8078282410685 ,
    #                    567.7671645871519 ,498.7547410153283 ,485.5944666179963 ,428.07612179503263 ,
    #                    388.6169930632023 ,386.18961037199415 ,376.08823042097316 ,338.8477133073723 ,
    #                    309.2480238229747 ,294.5568206195705 ,281.2171535075004 ,213.8208886916726 ,
    #                    254.72160898469483 ,195.57830703833136 ,211.93867859510462 ,199.35740205728564 ,
    #                    156.25468049264455 ,195.68094257648 ,145.5655485672349 ,142.54704777102097 ,
    #                    146.47263504779266 ,94.33540214158423 ,99.05286040252803 ,136.20629630937805 ,
    #                    100.99664099645533 ,69.68392363472498 ,62.14865572661666 ,52.90535288375964 ,
    #                    49.658854513180145 ,44.697987693731164 ,93.30094647357532 ,76.58788137313091 ,
    #                    44.495013750130134 ,49.78130435111957 ,28.591123274109403 ,14.369557430945626 ,
    #                    15.942197375234471 ,33.018212055985686 ,26.294114699100742 ,28.04829536743325 ,
    #                    37.51661668556541 ,24.48232853550999 ,31.99219938621985 ,52.34547007932935 ,
    #                    26.191496658272193 ,41.97686572440534 ,36.3838412672615 ,13.115824695742333 ,
    #                    23.5749116813857 ,40.302257429852816 ,15.969279873602586 ,7.895003815354128 ,
    #                    24.188839271486874 ,29.558535336818906 ,25.768383455514858 ,33.7149871310329 ,
    #                    33.41987503558679 ,28.325411074385094 ,28.696660350714314 ,14.07918430602742 ,
    #                    28.408626617687165 ,37.46161004997681 ,16.56374580286518 ,32.37433087637632 ,
    #                    9.729146991334474 ,31.138236631716335 ,9.167230200724529 ,18.04542095255966 ,
    #                    52.10527513423827 ,8.605375161904306 ,34.09528519417992 ,63.3642397915748 ,
    #                    61.82192674167916 ,10.402761307531772 ,43.507101594742785 ,17.706587093874955 ,
    #                    28.37424648150537 ,53.99808241916874 ,27.468625611884256 ,28.907958786532625 ,
    #                    5.016136842885782 ,12.331750562238897 ,12.152385742126548 ,14.708345988504204 ,
    #                    20.880343502584108 ,45.633794111081926 ,33.192065161666356 ,12.913596989980766 ,
    #                    45.29593334531513 ,29.53694349839379 ,46.163861173396285 ,30.7950601400292])



    #noiseFloorBreaker = hispeed.OneDimensionalSignalTransformationalFramework(bvalues.tolist(), noisySI.tolist(), 1, 20.)
   # noiseFloorBreaker = hispeed.OneDimensionalSignalTransformationalFramework([1000.3], [5.,6.1,6.,7.], args.N, 5.2)
    #noiseFloorBreaker.setProbabilityLevel(alpha)
    #noiseFloorBreaker.setDegreeAndKnots(degree, nknots)
    #noiseFloorBreaker.evaluate()
    #print(np.array(noiseFloorBreaker.getCorrectedNoisyY()))
    #1/0
    # noiseFloorBreaker.setProbabilityLevel(alpha_stab)
    # noiseFloorBreaker.setDegreeAndKnots(degree, nknots)
    # noiseFloorBreaker.evaluate()

    # print(sigma, time()-timea)
    # 1/0
    # arr = hispeed.JArray('object')(3)
    # arr[0] = hispeed.JArray('float')([10.])
    # arr[1] = hispeed.JArray('float')([10.])
    # arr[2] = hispeed.JArray('float')([1.])

    # print(data.shape, type(data), data.dtype, type(alpha), type(args.N))
    # sigma = hispeed.OptimalPIESNO([[5.], [1.], [6.]], alpha, args.N)#.getEstimatedGaussianNoiseSD()
    # print(sigma, time()-timea)
    # 1/0

    # from time import time
    # for i in range(data.shape[-1]):
    #     timea = time()
    #     #a= data[67:85,89:93,45:67, i].tolist()
    #     #data = (elem for iterable in a for elem in iterable)
    #     #print(data)
    #     # arr = hispeed.JArray('object')(3)
    #     # arr[0] = hispeed.JArray('float')([10.])
    #     # arr[1] = hispeed.JArray('float')([10.])
    #     # arr[2] = hispeed.JArray('float')([1.])
    #     # sigma[i] = hispeed.OptimalPIESNO(data, alpha, args.N).getEstimatedGaussianNoiseSD()
    #     # 1/0
    #     data_flat = []
    #     i, j = 0, 0
    #     for idx in range(data.ndim):
    #         i = j
    #         j = data.shape[idx] + i
    #         #data_flat[i:j] = data.ravel()[i:j]
    #         data_flat += data.ravel()[i:j].tolist()
    #     print (data.shape, len(data_flat), time()-timea)

    #     noiseFloorBreaker = hispeed.OneDimensionalSignalTransformationalFramework(bvals.tolist(), data_flat, args.N, sigma)

    #     noiseFloorBreaker.setProbabilityLevel(alpha)
    #     noiseFloorBreaker.setDegreeAndKnots(degree, nknots)
    #     noiseFloorBreaker.evaluate()

    #     data_stabilized[..., i] = np.array(noiseFloorBreaker.getCorrectedNoisyY()).reshape(data.shape[:-1])
    #     print(i, time() - timea)

    #print("Now stabilizing")
    #timea=time()
    #print(bvals.shape, data_flat.shape, bvals.dtype, data_flat.dtype, type(N), sigma.dtype)
    #noiseFloorBreaker = hispeed.OneDimensionalSignalTransformationalFramework(np.arange(100,dtype='float64').tolist(), np.arange(100,dtype='float64').tolist(), 1, float(sigma))
    #noiseFloorBreaker = hispeed.OneDimensionalSignalTransformationalFramework(64*[1000], data[70,70,30,1:].ravel().tolist(), N, sigma)
    #noiseFloorBreaker = hispeed.OneDimensionalSignalTransformationalFramework((np.ones_like(data_flat[:100])*1000).tolist(), data_flat[:100].tolist(), N, sigma)
    #print("pass!")
    #1/0
    #noiseFloorBreaker.setProbabilityLevel(alpha)
    #noiseFloorBreaker.setDegreeAndKnots(degree, nknots)
    #noiseFloorBreaker.evaluate()

    #data_gauss = hispeed.nonCentralChiToGaussian(data_flat, eta, sigma, N, alpha)
    #print("finished, time was", time()-timea)
    #data_stabilized = noiseFloorBreaker.getCorrectedNoisyY().reshape(o_shape).astype(o_dtype)


if __name__ == "__main__":
    main()
