#!/usr/bin/python
# coding:utf-8

import argparse
import code
import Image
import pylab as pl

import flattnr.image_analysis as img_an


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('--column', type=int, default=75)
    args = parser.parse_args()

    aa = Image.open(args.input).convert('L')
    image = pl.array(aa, dtype=pl.np.float)

    spacing = img_an.estimate_general_line_spacing(image)
    sf = 4.0 / spacing
    print "Estimated line spacing: {}".format(spacing)
    print "Scaling factor: {}".format(sf)

    qq, lpqq, hpqq = img_an.scale_and_filter_image(image, sf)

    mask_l, centroid_l, mask_r, centroid_r = img_an.get_page_masks(hpqq)

    pl.ion()

    d = 10
    step = 10

    all_edgels = img_an.extract_edgels(hpqq, d, step)

    edgels = [(p, v, edgel_intensity) for p, v, edgel_intensity in all_edgels
              if mask_l[p[1], p[0]] and edgel_intensity > 1.5]

    plot_edgels(qq, edgels, 6.0)


def plot_edgels(img, edgels, s):
    # Plot edgels and
    xx = pl.array([p for p, _, _ in edgels])
    pl.figure(figsize=(12, 6))
    pl.imshow(img, cmap=pl.cm.gray, interpolation='nearest')
    pl.axis('equal')
    pl.plot(xx[:, 0], xx[:, 1], 'r.')
    for p, v, ei in edgels:
        xx, yy = pl.c_[p - s * v, p + s * v]
        pl.plot(xx, yy, 'r-')
    code.interact()


if __name__ == '__main__':
    main()
