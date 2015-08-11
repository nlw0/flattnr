#!/usr/bin/python2.7
# coding: utf-8
import argparse
import Image
from pylab import *
import scipy
import scipy.signal
import scipy.ndimage.morphology as morph
import scipy.signal

import numpy.linalg
import numpy

import code

from scipy.linalg import toeplitz


def get_page_masks(hpqq):
    rd = osc_detector(hpqq)

    rd = morph.binary_dilation(rd, iterations=3)
    rd = morph.binary_erosion(rd, iterations=5)

    label_im, nb_labels = scipy.ndimage.label(rd)

    sizes = scipy.ndimage.sum(rd, label_im, range(nb_labels + 1))

    mask_size = sizes < (sort(sizes)[-2])
    mask_l = morph.binary_fill_holes(label_im == nonzero(sizes == sort(sizes)[-1])[0][0])
    mask_r = morph.binary_fill_holes(label_im == nonzero(sizes == sort(sizes)[-2])[0][0])

    yy, xx = mgrid[:hpqq.shape[0], :hpqq.shape[1]]

    centroid_l = ((mask_l * xx).sum() / mask_l.sum(),
                  (mask_l * yy).sum() / mask_l.sum())

    centroid_r = ((mask_r * xx).sum() / mask_r.sum(),
                  (mask_r * yy).sum() / mask_r.sum())

    ## Make sure the "left" actually is the one we assumed.
    if centroid_l[0] > centroid_r[0]:
        mask_l, mask_r = mask_r, mask_l
        centroid_l, centroid_r = centroid_r, centroid_l

    return mask_l, centroid_l, mask_r, centroid_r


def osc_detector(img):
    return array(
        [osc_detector_single_line(col)
         for col in img.T]
    ).T


def osc_detector_single_line(line):
    Nw = 5
    N = 2 * Nw + 1

    # Lanczos window
    lanczos = sinc(mgrid[-Nw:1 + Nw] * (1.0 / (Nw + 1.0)))

    ss = toeplitz(line[N - 1:], line[N - 1::-1])
    ss = dot(ss, diag(lanczos))

    M = array([
        [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
    ])

    M = dot(diag(1 / sqrt(numpy.sum(M ** 2, 1))), M)

    qq = dot(ss, M.T)
    qq = sqrt(qq[:, 0] ** 2 + qq[:, 1] ** 2)
    ww = sqrt((ss ** 2).sum(1))
    sel = (qq > 5.0) * (qq / ww > 0.55)

    return r_[zeros(N / 2), sel, zeros(N / 2)]


def quadratic_surface(N, a):
    x = mgrid[:N] - ((N - 1) * .5)
    X = repeat(x, N, 0).reshape(-1, N).T
    XX = X * X
    XY = X * X.T
    YY = XX.T

    return a[0] * XX + a[1] * XY + a[2] * YY


def extract_edgels(hpqq, d, step):
    return [
        (array([cx, cy]),) + estimate_texture_direction(fetch_patch(hpqq, cx, cy, d))
        for cx in mgrid[d - 1:hpqq.shape[1] - d:step]
        for cy in mgrid[d - 1:hpqq.shape[0] - d:step]]


def fetch_patch(hpqq, cx, cy, d):
    return hpqq[cy - d + 1: cy + d, cx - d + 1: cx + d]


def estimate_texture_direction(patch):
    period = patch.shape[0]
    ww = scipy.signal.gaussian(period, period / 6.0)
    wpatch = patch * outer(ww, ww)
    QQ = patch_autocorr(wpatch)

    B = QQ.ravel()

    A = array([[0.33333333, 1., 0.33333333],
               [-0.66666667, 0., 0.33333333],
               [0.33333333, -1., 0.33333333],
               [0.33333333, 0., -0.66666667],
               [-0.66666667, 0., -0.66666667],
               [0.33333333, 0., -0.66666667],
               [0.33333333, -1., 0.33333333],
               [-0.66666667, 0., 0.33333333],
               [0.33333333, 1., 0.33333333]])

    a = numpy.linalg.lstsq(A, B)[0]

    M = array([[a[0], a[1] / 2],
               [a[1] / 2, a[2]]])

    U, S, V = svd(M)

    q = V[1]
    q = q * sign(q[0])

    return q, wpatch.std()


def patch_autocorr(xx):
    XX = fft2(xx)
    ac = real(ifft2(XX * conj(XX)))

    QQ = zeros((3, 3))
    QQ[1:, 1:] = ac[:2, :2]
    QQ[0, 0] = ac[-1, -1]
    QQ[0, 1:] = ac[-1, :2]
    QQ[1:, 0] = ac[:2, -1]

    return QQ


def find_first_peak(x):
    k = 1
    while not (x[k] > x[k - 1] and x[k] >= x[k + 1]):
        k += 1
    return k


def estimate_general_line_spacing(image):
    ac = calculate_image_vertical_autocorrelation(image)
    return find_first_peak(ac)


def calculate_image_vertical_autocorrelation(image):
    _max_exp_line_count = 10
    sigma = min(image.shape) / (_max_exp_line_count * 6.0)

    lowpass = scipy.ndimage.gaussian_filter(image, sigma)
    highpass = image - lowpass

    ww = repeat([hamming(image.shape[0])], image.shape[1], 0)

    RR = fft(ww * highpass.T)
    ac = real(ifft((RR * conj(RR)).sum(0)))
    return ac


def scale_and_filter_image(image, scale_factor, lpf_param=2.0):
    """
    Apply anti-alias filter, and scale down so the line spacing is
    approximately 4 pixels.

    
    The 2.0 parameter cuts the oscillation from the text lines while
    leaving the filter "local" enough. Increasing it makes the book
    borders too wide in the high-pass filtered image. There must be a
    better way to do this filtering, but for the moment this looks
    good enough.

    """

    image_antialiased = scipy.ndimage.gaussian_filter(image, 0.5 / scale_factor)
    scaled_image = scipy.ndimage.zoom(image_antialiased, scale_factor, order=5)
    lpf_scaled_image = scipy.ndimage.gaussian_filter(scaled_image, lpf_param)
    hpf_scaled_image = scaled_image - lpf_scaled_image
    return scaled_image, lpf_scaled_image, hpf_scaled_image


def plot_samples(ax, signal):
    yy = c_[zeros(len(hpqq)), signal]
    xx = c_[range(len(hpqq)), range(len(hpqq))]
    ax.plot(xx.T, yy.T, 'b-')
    ax.plot(signal, '.')
