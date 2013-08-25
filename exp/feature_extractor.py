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


def ridge_detector(img):
    zzaa = array([1,1,-1,-1,1,1,-1,-1.0])
    zzbb = array([-1,1,1,-1,-1,1,1,-1.0])
    zza = array([1,1,1,1,-1,-1,-1,-1.0])
    zzb = array([-1,-1,1,1,1,1,-1,-1.0])
    # zza = array([-1, 0, 1, 0,-1.0])
    # zzb = array([ 0,-1, 0, 1, 0.0])
    a = array(
        [convolve(zza, col)
         for col in img.T]).T
    b = array(
        [convolve(zzb, col)
         for col in img.T]).T
    aa = array(
        [convolve(zzaa, col)
         for col in img.T]).T
    bb = array(
        [convolve(zzbb, col)
         for col in img.T]).T

    return a, b, aa, bb

def quadratic_surface(N, a):
    x = mgrid[:N] - ((N-1) * .5)
    X = repeat(x, N, 0).reshape(-1,N).T
    XX = X * X
    XY = X * X.T
    YY = XX.T

    return a[0] * XX + a[1] * XY + a[2] * YY

def extract_edgels(hpqq, d):
    step = d / 3
    return [
        (array([cx, cy]),) + estimate_texture_direction(fetch_patch(hpqq, cx, cy, d))
        for cx in mgrid[d-1:hpqq.shape[1] - d:step]
        for cy in mgrid[d-1:hpqq.shape[0] - d:step]]

def fetch_patch(hpqq, cx, cy, d):
    return hpqq[cy-d+1:cy+d, cx-d+1:cx+d]

def estimate_texture_direction(patch):
    period = patch.shape[0]
    # ww = hamming(period)
    # ww = scipy.signal.gaussian(period, period/6.0)
    ww = scipy.signal.gaussian(period, period/6.0)
    wpatch = patch * outer(ww, ww)

    QQ = patch_autocorr(wpatch)

    XX = array([[1, 0, 1], [1, 0, 1], [ 1, 0, 1]]) - 2 / 3.
    XY = array([[1, 0,-1], [0, 0, 0], [-1, 0, 1]])
    YY = array([[1, 1, 1], [0, 0, 0], [ 1, 1, 1]]) - 2 / 3.

    A = zeros((9, 3))
    B = zeros(9)

    A[:,0] = XX.ravel()
    A[:,1] = XY.ravel()
    A[:,2] = YY.ravel()
    B = QQ.ravel()

    a = numpy.linalg.lstsq(A, B)[0]

    M = array([[a[0], a[1]/2],
               [a[1]/2, a[2]]])

    U, S, V = svd(M)

    q = V[1]
    q = q * sign(q[0])

    return q, wpatch.std()

def patch_autocorr(xx):
    XX = fft2(xx)
    ac = real(ifft2(XX * conj(XX)))

    QQ = zeros((3,3))
    QQ[1:,1:] = ac[:2,:2]
    QQ[0,0] = ac[-1,-1] 
    QQ[0,1:] = ac[-1,:2]
    QQ[1:,0] = ac[:2,-1]

    return QQ

def find_first_peak(x):
    k = 1
    while not (x[k] > x[k-1] and x[k] >= x[k+1]):
        k += 1
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

def scale_and_filter_image(image, scale_factor, lpf_param=2.0):
    '''
    Apply anti-alias filter, and scale down so the line spacing is
    approximately 4 pixels.

    
    The 2.0 parameter cuts the oscilation form the text lines while
    leaving the filter "local" enough. Increasing it makes the book
    borders too wide in the high-pass filtered image. There must be a
    better way to do this filtering, but for the moment this looks
    good enough.

    '''

    image_antiali = scipy.ndimage.gaussian_filter(image, 0.5 / scale_factor)
    scaled_image = scipy.ndimage.zoom(image_antiali, scale_factor, order=5)
    lpf_scaled_image = scipy.ndimage.gaussian_filter(scaled_image, lpf_param)
    hpf_scaled_image = scaled_image - lpf_scaled_image
    return scaled_image, lpf_scaled_image, hpf_scaled_image

def plot_samples(ax, signal):
    yy = c_[zeros(len(hpqq)), signal]
    xx = c_[range(len(hpqq)), range(len(hpqq))]
    ax.plot(xx.T, yy.T, 'b-')
    ax.plot(signal, '.')


if __name__ == '__main__':

    #generate teal-and-orange colormap with 1024 interpolated values
    cdict = {
        'red'   :  ((0., 0., 0.00), (0.2, 0.30, 0.30), (0.8, 1.00, 1.00), (1., 1.00, 1.)),
        'green' :  ((0., 0., 0.25), (0.2, 0.47, 0.47), (0.8, 0.82, 0.82), (1., 0.75, 1.)),
        'blue'  :  ((0., 0., 1.00), (0.2, 1.00, 1.00), (0.8, 0.30, 0.30), (1., 0.00, 1.)),
        }
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('--column', type=int, default=75)
    args = parser.parse_args()

    aa = Image.open(args.input).convert('L')
    image = array(aa, dtype=np.float)

    spacing = estimate_general_line_spacing(image)
    sf = 4.0 / spacing
    print "Estimated line spacing: {}".format(spacing)
    print "Scaling factor: {}".format(sf)

    qq, lpqq, hpqq = scale_and_filter_image(image, sf)

    ion()


    cc = args.column
    

    ##
    ##
    figure(figsize=(9, 9))
    suptitle('Plots from column number {}'.format(cc))
    ax = subplot(3,1,1)
    title('Original image')
    plot(qq[:,cc], 'b-', label='Original')
    plot(lpqq[:,cc], 'r:', label='Low-pass')
    legend(loc='lower center', ncol=2)

    subplot(3,1,2, sharex=ax, sharey=ax)
    title('Low-pass filtered')
    plot(qq[:,cc], 'b:+', label='Original')
    plot(lpqq[:,cc], 'r-', label='Low-pass')
    legend(loc='lower center', ncol=2)

    bx = subplot(3,1,3, sharex=ax)
    title('High-pass filtered')
    # plot_samples(bx, hpqq[:,cc])
    plot(hpqq[:,cc], 'b-', label='High-pass')
    plot([0, len(qq)], [0,0], 'k--')
    legend(loc='lower center')
    

    ##
    ##
    figure(figsize=(12.5, 9))
    ax = subplot(2,2,1)
    title('Scaled book picture')
    imshow(qq, cmap=cm.gray)
    plot([cc, cc], [0, len(qq)], 'k:')

    subplot(2,2,2, sharex=ax, sharey=ax)
    title('Low-pass filtered')
    imshow(lpqq, cmap=cm.gray)
    plot([cc, cc], [0, len(qq)], 'k:')

    subplot(2,2,4, sharex=ax, sharey=ax)
    imshow(hpqq, vmin=-20, vmax=20, cmap=my_cmap)
    title('High-pass filtered')
    axis_store = axis()
    plot([cc, cc], [0, len(qq)], 'k:')
    tight_layout()
    axis(axis_store)

    subplot(2,2,3, sharex=ax, sharey=ax)
    title('Estimated text line directions')

    d = 10
    s = 5.0

    edgels = extract_edgels(hpqq, d)

    oo = array([1.0, 1.0])
    for p, v, edgel_intensity in edgels:
        if edgel_intensity > 0:#1.5:
            xx, yy = c_[p - s*v, p + s*v]
            plot(xx, yy, 'r-')

    ang = zeros((100,100,2))
    for p, v, edgel_intensity in edgels:
        if edgel_intensity <= 1.5 or v[0] < 0.9:
            continue
        x, y = p / (d / 3)
        ang[y, x, :] = v

    figure()
    imshow(ang[:,:,1], vmin=-0.4, vmax=0.4)
