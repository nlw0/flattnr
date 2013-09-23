#!/usr/bin/python
#coding:utf-8

import argparse
import code 
import Image
from matplotlib import colors
from numpy import array, reshape, c_, mgrid, pi, sin, cos, dot
from numpy.linalg import det, norm
import pylab

import flattnr.image_analysis as img_an
    
def main():
    #generate teal-and-orange colormap with 1024 interpolated values
    cdict = {
        'red'   :  ((0., 0., 0.00), (0.2, 0.30, 0.30), (0.8, 1.00, 1.00), (1., 1.00, 1.)),
        'green' :  ((0., 0., 0.25), (0.2, 0.47, 0.47), (0.8, 0.82, 0.82), (1., 0.75, 1.)),
        'blue'  :  ((0., 0., 1.00), (0.2, 1.00, 1.00), (0.8, 0.30, 0.30), (1., 0.00, 1.)),
        }
    my_cmap = pylab.matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    ## Command-line argument parsing
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('--column', type=int, default=75)
    args = parser.parse_args()

    aa = Image.open(args.input).convert('L')
    image = array(aa, dtype=pylab.np.float)

    spacing = img_an.estimate_general_line_spacing(image)
    sf = 4.0 / spacing
    print "Estimated line spacing: {}".format(spacing)
    print "Scaling factor: {}".format(sf)

    qq, lpqq, hpqq = img_an.scale_and_filter_image(image, sf)

    mask_l, centroid_l, mask_r, centroid_r = img_an.get_page_masks(hpqq)


    pylab.ion()
    cc = args.column

    
    from pylab import *
    figure(figsize=(6,9))
    suptitle('Text area detection')
    ax = subplot(211)
    imshow(hpqq)
    subplot(212, sharex=ax, sharey=ax)
    imshow(1*mask_l - mask_r)

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

    edgels = img_an.extract_edgels(hpqq, d)

    oo = array([1.0, 1.0])
    for p, v, edgel_intensity in edgels:
        # if edgel_intensity > 0:#1.5:
        if mask_l[p[1], p[0]]:
            xx, yy = c_[p - s*v, p + s*v]
            plot(xx, yy, 'r-')

    ang_l = zeros((60,100,2))
    ang_r = zeros((60,100,2))
    for p, v, edgel_intensity in edgels:
        if edgel_intensity <= 1.5 or v[0] < 0.9:
            continue
        x, y = p / (d / 3)
        if mask_l[p[1], p[0]]:
            ang_l[y, x, :] = v
        elif mask_r[p[1], p[0]]:
            ang_r[y, x, :] = v

    figure(figsize=(12,6))
    ax = axes([0.02,0.02,0.2,0.95])
    imshow(ang_l[:,:,1], vmin=-0.4, vmax=0.4, extent=[0, qq.shape[1], qq.shape[0], 0])
    axis('equal')
    axes([0.78,0.02,0.2,0.95], sharey=ax)
    imshow(ang_r[:,:,1], vmin=-0.4, vmax=0.4, extent=[0, qq.shape[1], qq.shape[0], 0])
    axis('equal')
    axes([0.25,0.02,0.51,0.95], sharey=ax)
    imshow(qq, cmap=cm.gray)
    axis('equal')


    code.interact()

if __name__ == '__main__':
    main()
