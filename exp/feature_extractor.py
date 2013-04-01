#!/usr/bin/python2.7

import argparse
import Image
from pylab import *
import scipy
import scipy.signal
import scipy.ndimage.morphology as morph

import code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('input', type=argparse.FileType('r'))
    args = parser.parse_args()

    aa = Image.open(args.input)
    qq = array(aa, dtype=np.float)

    ss = scipy.ndimage.gaussian_filter(qq, 3.0)
    
    #cx,cy = 470,100
    #cx,cy = 300,200
    cx,cy = 150,530
    period = 15
    d = floor(2.0 * period)
    patch = copy(ss[cy-d:cy+d+1, cx-d:cx+d+1])

    patch -= patch.mean()
    gau = scipy.signal.gaussian(2 * d + 1, (2 * d + 1) / 6.0)
    Wg = outer(gau, gau)
    patch = patch * Wg

    PP = fft2(patch)
    RR = fftshift(real(ifft2(PP * conj(PP))))



    ion()
    figure()
    imshow(qq, cmap=cm.bone)

    figure()
    imshow(ss, cmap=cm.bone)

    figure()
    imshow(patch)

    figure()
    imshow(RR, extent=(-d-.5, d+.5, -d-.5, d+.5))

#    code.interact(local=locals())



