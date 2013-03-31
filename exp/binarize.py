#!/usr/bin/python2.7

import argparse
import Image
from pylab import *
import scipy.ndimage.morphology as morph

import code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('input', type=argparse.FileType('r'))
    args = parser.parse_args()

    aa = Image.open(args.input)
    qq = array(aa, dtype=np.float)

    ww = morph.grey_opening(qq, size=(7,7))

    ion()
    ax = subplot(1,2,1)
    imshow(qq, cmap=cm.bone)
    subplot(1,2,2, sharex=ax, sharey=ax)
    imshow(ww, cmap=cm.bone)

    code.interact(local=locals())



