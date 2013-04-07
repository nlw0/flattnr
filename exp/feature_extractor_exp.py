#!/usr/bin/python2.7

import argparse
import Image
from pylab import *
import scipy
import scipy.signal
import scipy.ndimage.morphology as morph

import numpy.linalg

import code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('input', type=argparse.FileType('r'))
    args = parser.parse_args()

    aa = Image.open(args.input)
    qq = array(aa, dtype=np.float)

    ss = scipy.ndimage.gaussian_filter(qq, 3.0)
    
    #cx,cy = 120,500
    #cx,cy = 470,110
    cx,cy = 460,100
    #cx,cy = 450,140
    #cx,cy = 300,200
    #cx,cy = 390,95
    #cx,cy = 350,200
    #cx,cy = 150,530
    #cx,cy = 360,160
    
    period = 15
    d = floor(2.0 * period)
    patch = copy(ss[cy-d:cy+d+1, cx-d:cx+d+1])

    patch -= patch.mean()
    gau = scipy.signal.gaussian(2 * d + 1, (2 * d + 1) / 6.0)
    Wg = outer(gau, gau)
    patch = patch * Wg

    PP = fft2(patch)
    #RR = fftshift(real(ifft2(PP * conj(PP))))
    RR = real(ifft2(PP * conj(PP)))
    hd = period/4 + 1
    QQ = c_[RR[:hd,-hd+1:], RR[:hd,:hd]]
    QQ = r_[c_[RR[-hd+1:,-hd+1:], RR[-hd+1:,:hd]], QQ]
    QQ -= QQ.mean()

    x = mgrid[-hd+1:hd]
    X = repeat(x, hd*2-1,0).reshape(-1,2*hd-1).T
    XX = X * X
    XY = X * X.T
    YY = XX.T

    XX = XX - XX.mean()
    XY = XY - XY.mean()
    YY = YY - YY.mean()

    A = zeros(((2 * hd - 1) ** 2, 3))
    B = zeros((2 * hd - 1) ** 2)

    A[:,0] = XX.ravel()
    A[:,1] = XY.ravel()
    A[:,2] = YY.ravel()
    B = QQ.ravel()

    a = numpy.linalg.lstsq(A, B)[0]

    M = array([[a[0], a[1]/2],
               [a[1]/2, a[2]]])

    u,s,v = svd(M)

    dd = v[1]

    print a
    print M
    print v[0]

    print x

    ion()
    figure()
    imshow(qq, cmap=cm.bone)

    s = hd*4
    plot(array([cx-s*dd[0],cx+s*dd[0]]),
         array([cy-s*dd[1],cy+s*dd[1]]), 'r-')

    figure()
    imshow(ss, cmap=cm.bone)

    figure()
    imshow(patch)

    s = 30
    plot(s*array([1-dd[0],1+dd[0]]),
         s*array([1-dd[1],1+dd[1]]), 'r-')
    plot(s*array([1-dd[1],1+dd[1]]),
         s*array([1+dd[0],1-dd[0]]), 'r-')


    figure()
    #imshow(RR, extent=(-d-.5, d+.5, -d-.5, d+.5))
    imshow(QQ)



    figure()
    imshow(a[0] * XX + a[1] * XY + a[2] * YY)

    s = 3
    plot(s*array([1-dd[0],1+dd[0]]),
         s*array([1-dd[1],1+dd[1]]), 'r-')
    plot(s*array([1-dd[1],1+dd[1]]),
         s*array([1+dd[0],1-dd[0]]), 'r-')

#    code.interact(local=locals())
