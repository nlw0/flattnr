#!/usr/bin/python
#coding:utf-8

import code 

import flattnr.mesh

import argparse

from numpy import array, reshape, c_, mgrid, pi, sin, cos, dot
from numpy.linalg import det, norm

import pylab

from flattnr.quaternion import Quat

def plot_2d_mesh(ax, pp, Nrows, Ncols, color='b'):
    mesh = pp.reshape(Nrows, Ncols, -1)
    for pp in mesh:
        ax.plot(pp[:,0], pp[:,1], '-', color=color)
    for pp in [mesh[:,k,:] for k in xrange(mesh.shape[1])]:
        ax.plot(pp[:,0], pp[:,1], '-', color=color)
    pp = reshape(mesh, (-1,2))
    ax.plot(pp[:,0], pp[:,1], 'o', color=color)

def test_label():
    pass

def setup_mesh():
    mesh_rows = 9
    mesh_cols = 7
    mesh = flattnr.mesh.Mesh(mesh_rows, mesh_cols)

    angle = (pi * 2 / (mesh_cols-1)) / 5
    tt = angle * (mgrid[0.0 : mesh_cols] - mesh_cols / 2)
    circ = c_[sin(tt), -cos(tt)]

    for ii in range(mesh.Nrows):
        mesh.mesh[ii, :, 0] = circ[:,0]
        mesh.mesh[ii, :, 2] = circ[:,1] + 1
        mesh.mesh[ii, :, 1] = (ii - mesh.Nrows / 2) * angle

    M = Quat(1.0,0.08,-0.1,-0.1).normalize().rot()
    #M = Quat(1.0,0,0,0,).normalize().rot()

    mesh.points[:, :] = dot(mesh.points, M.T)
    mesh.points[:, 0] -= 0.2
    mesh.points[:, 1] += 0.0
    mesh.points[:, 2] += 1.9

    return mesh
    
def main():
    ## Command-line argument parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('command', type=str, choices=['labels', 'reverse', 'derivative', 'textlines', 'nada'])
    parser.add_argument('--columns', default=False, action='store_true')
    parser.add_argument('--reverse', default=False, action='store_true')

    args = parser.parse_args()

    mesh = setup_mesh()

    cm = flattnr.mesh.CameraModel(array([750, 1000])/4, array([500,375.0])/4, 750.0/4)
    pp = cm.project(mesh.points)
    if args.command == 'labels':
        ww = mesh.label_pixels(cm)

    if args.command == 'textlines':
        if not args.columns:
            uv = mgrid[-0.5:6.5:0.01,-0.1:8.2:0.1].T.reshape(-1,2)
        else:
            uv = mgrid[-0.5:6.5:0.1,-0.1:8.2:0.01].T.reshape(-1,2)
        xyz = array([mesh.uv_to_xyz(x) for x in uv])
        pq = array([mesh.uv_to_pq(cm, x) for x in uv])

    if args.command == 'derivative':
        if args.reverse:
            pq = mgrid[35:175:10,15:185:10].T.reshape(-1,2)
            uv = array([mesh.pq_to_uv(cm, ipq) for ipq in pq])
            xyz = array([mesh.uv_to_xyz(iuv) for iuv in uv])
        else:
            uv = mgrid[-0.5:6.5:0.3,-0.1:8.2:0.3].T.reshape(-1,2)
            xyz = array([mesh.uv_to_xyz(x) for x in uv])
            pq = array([mesh.uv_to_pq(cm, x) for x in uv])

        mesh.calculate_derivative()

    if args.command == 'reverse':
        Xpq = mgrid[25:175:25,25:175:25].T.reshape(-1,2)
        Xuv = array([mesh.pq_to_uv(cm, pq) for pq in Xpq])
        Xxyz = array([mesh.uv_to_xyz(uv) for uv in Xuv])

    fig = pylab.figure(figsize=(9,9))
    ax = mesh.plot_wireframe()
    ax.set_title('Book page mesh model - 3D')
    if args.command == 'reverse':
        for line in Xxyz:
            lx, ly, lz = c_[array([0.0,0.0,0.0]), 1.2 * line]
            ax.plot(lx,lz,ly, 'r-', alpha=0.5)
        ax.plot(Xxyz[:,0], Xxyz[:,2], Xxyz[:,1], 'ro')
    if args.command == 'textlines':
        pylab.plot(xyz[:,0], xyz[:,2], xyz[:,1],  'k,')
    if args.command == 'derivative':
        pylab.plot(xyz[:,0], xyz[:,2], xyz[:,1],  'k.')

        for iuv, ixyz in zip(uv, xyz):
            dd = ixyz + 0.25 * mesh.uv_to_xyz_dev(iuv)
            pylab.plot(array([ixyz[0], dd[0]]),
                       array([ixyz[2], dd[2]]),
                       array([ixyz[1], dd[1]]), 'k-'
                    )
    cm.draw_camera(ax)
    ax.plot([0],[0],[0], 'ko')  # Focal point


    fig = pylab.figure()
    ax = pylab.subplot(1,1,1)    
    ax.set_title('Book page mesh model projection')
    if args.command == 'labels':
        pylab.imshow(ww)
    plot_2d_mesh(ax, pp, mesh.Nrows, mesh.Ncols)
    ax.axis('equal')
    pylab.axis([0,cm.image_shape[1], cm.image_shape[0],0])
    if args.command == 'textlines':
        pylab.plot(pq[:,0], pq[:,1], 'k,')
    if args.command == 'derivative':
        pylab.plot(pq[:,0], pq[:,1], 'k.')

        J = cm.jacobian_from_pq(pq)

        for iuv, ixyz, ipq, iJ in zip(uv, xyz, pq, J):
            dd = dot(iJ, mesh.uv_to_xyz_dev(iuv))
            dd = dd / norm(dd)

            dd = ipq + 5 * dd
            pylab.plot([ipq[0], dd[0]],
                       [ipq[1], dd[1]], 'k-'
                    )
    if args.command == 'reverse':
        ax.plot(Xpq[:,0], Xpq[:,1], 'rx')
        plot_2d_mesh(ax, Xpq, 6, 6, 'r')
    ax.grid()

    if args.command == 'reverse':
        fig = pylab.figure()
        ax = pylab.subplot(1,1,1)    
        ax.set_title('Image mesh projected into texture space.')
        ax.plot(Xuv[:,0], Xuv[:,1], 'rx')
        plot_2d_mesh(ax, Xuv, 6, 6, 'r')
        ax.axis('equal')
        ax.grid()

    if args.command == 'textlines':
        fig = pylab.figure()
        ax = pylab.subplot(1,1,1)    
        ax.set_title('Texture space')
        plot_2d_mesh(ax,
                     mgrid[0:mesh.Ncols, 0:mesh.Nrows].T.reshape(-1,2),
                     mesh.Nrows, mesh.Ncols, 'b')
        ax.plot(uv[:,0], uv[:,1], 'k,')
        ax.axis('equal')
        ax.set_ylim(8.5,-.5)
        ax.grid()

    if args.command == 'derivative':
        pylab.figure()
        pylab.imshow(mesh.mesh[:,:,2])
        pylab.figure()
        pylab.imshow(mesh.mesh_dev[:,:,2])

    import ipdb; ipdb.set_trace();



if __name__ == '__main__':
    main()

