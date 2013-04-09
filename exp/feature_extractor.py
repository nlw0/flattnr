#!/usr/bin/python2.7
#coding: utf-8
import argparse
import Image
from pylab import *
import scipy
import scipy.signal
import scipy.ndimage.morphology as morph
import scipy.signal

import numpy.linalg

import code



def quadratic_surface(N, a):
    x = mgrid[:N] - ((N-1) * .5)
    X = repeat(x, N, 0).reshape(-1,N).T
    XX = X * X
    XY = X * X.T
    YY = XX.T

    return a[0] * XX + a[1] * XY + a[2] * YY


def patch_autocorr(xx):
    XX = fft2(xx)
    ac = real(ifft2(XX * conj(XX)))
    return ac


def estimate_texture_direction(patch):
    period = patch.shape[0]
    # ww = hamming(period)
    # ww = scipy.signal.gaussian(period, period/6.0)
    ww = scipy.signal.gaussian(period, period/6.0)
    wpatch = patch * outer(ww, ww)

    ac = patch_autocorr(wpatch)

    QQ = zeros((3,3))
    QQ[1:,1:] = ac[:2,:2]
    QQ[0,0] = ac[-1,-1] 
    QQ[0,1:] = ac[-1,:2]
    QQ[1:,0] = ac[:2,-1]

    XX = array([[1, 0, 1], [1, 0, 1], [1, 0, 1]]) - 2 / 3.
    XY = array([[ 1, 0,-1], [ 0, 0, 0], [-1, 0, 1]])
    YY = array([[1, 1, 1], [0, 0, 0], [1, 1, 1]]) - 2 / 3.

    A = zeros((9, 3))
    B = zeros(9)

    A[:,0] = XX.ravel()
    A[:,1] = XY.ravel()
    A[:,2] = YY.ravel()
    B = QQ.ravel()

    a = numpy.linalg.lstsq(A, B)[0]

    M = array([[a[0], a[1]/2],
               [a[1]/2, a[2]]])

    U,S,V = svd(M)

    q = V[1]

    return q, wpatch.std()

def find_first_peak(x):
    k = 0
    while True:
        k += 1
        if x[k-1] < x[k] and x[k] > x[k+1]:
            return k
        

def estimate_general_line_spacing(image):
    ac = calculate_image_vertical_autocorrelation(image)
    return find_first_peak(ac)


def calculate_image_vertical_autocorrelation(image):
    _max_exp_line_count = 10
    # sigma = min(image.shape) / (_max_exp_line_count * 6.0)
    sigma = min(image.shape) / (_max_exp_line_count * 6.0)

    lowpass = scipy.ndimage.gaussian_filter(image, sigma)
    highpass = image - lowpass

    ww = repeat([hamming(image.shape[0])], image.shape[1], 0)

    RR = fft(ww * highpass.T)
    ac = real(ifft((RR * conj(RR)).sum(0)))
    return ac


if __name__ == '__main__':

    cdict = {
        'red'   :  ((0., 0., 0.00), (0.2, 0.30, 0.30), (0.8, 1.00, 1.00), (1., 1.00, 1.)),
        'green' :  ((0., 0., 0.25), (0.2, 0.47, 0.47), (0.8, 0.82, 0.82), (1., 0.75, 1.)),
        'blue'  :  ((0., 0., 1.00), (0.2, 1.00, 1.00), (0.8, 0.30, 0.30), (1., 0.00, 1.)),
        }


# 0.00 0.25 1.0
# 0.30 0.47 1.0
# 1.0 0.82 0.30
# 1.0 0.75 0.0

    #generate the colormap with 1024 interpolated values
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)



    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('input', type=argparse.FileType('r'))
    args = parser.parse_args()

    aa = Image.open(args.input).convert('L')
    image = array(aa, dtype=np.float)

    spacing = estimate_general_line_spacing(image)
    sf = 4.0 / spacing
    print "Estimated line spacing: {}".format(spacing)
    print "Scaling factor: {}".format(sf)

    image_aali = scipy.ndimage.gaussian_filter(image, 0.5 / sf)
    qq = scipy.ndimage.zoom(image_aali, sf, order=5)

    lpqq = scipy.ndimage.gaussian_filter(qq, 2.0)
    hpqq = qq - lpqq

    ion()


    figure()
    imshow(qq, cmap=cm.gray)

    figure()
    imshow(lpqq, cmap=cm.gray)

    figure()
    imshow(hpqq, vmin=-20, vmax=20, cmap=my_cmap)

    figure(figsize=(7,10))
    subplot(2,1,1)
    title('Scaled book picture')
    imshow(qq, cmap=cm.gray)
    subplot(2,1,2, sharex=gca(), sharey=gca())
    title('Estimated text line directions')
    #imshow(-abs(hpqq), cmap=cm.bone)
    
    # figure(3)


    d = 10
    s = 5.0
    
    Lss = []

    for cx in mgrid[d:hpqq.shape[1]-d:5]:
        for cy in mgrid[d:hpqq.shape[0]-d:5]:
            #cx, cy = 13,107
            patch = hpqq[cy-d+1:cy+d, cx-d+1:cx+d]
            q, ss = estimate_texture_direction(patch)
            Lss.append(ss)

            if ss < 1.5:
                continue
            plot(cx+s*array([-q[0], q[0]]),
                 cy+s*array([-q[1], q[1]]), 'r-')


    xlim(0,qq.shape[1])
    ylim(qq.shape[0],0)

    savefig('out.png')


    # figure()
    # plot(sort(Lss))
#    code.interact(local=locals())
