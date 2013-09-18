#!/usr/bin/python
#coding:utf-8

import code 

import flattnr.mesh

import argparse

from numpy import array, reshape, c_, mgrid, pi, sin, cos, dot
from numpy.linalg import det

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

def main():
    ## Command-line argument parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_file', type=str)

    args = parser.parse_args()

    mesh_lins = 9
    mesh_cols = 7
    mesh = flattnr.mesh.Mesh(mesh_lins, mesh_cols)

    angle = (pi * 2 / (mesh_cols-1)) / 5
    tt = angle * (mgrid[0.0 : mesh_cols] - mesh_cols / 2)
    circ = c_[sin(tt), -cos(tt)]

    for ii in range(mesh_lins):
        mesh.mesh[ii, :, 0] = circ[:,0]
        mesh.mesh[ii, :, 2] = circ[:,1] + 1
        mesh.mesh[ii, :, 1] = (mesh_cols/2 - ii) * angle

    M = Quat(1.0,0.08,-0.1,-0.1).normalize().rot()
    #M = Quat(1.0,0,0,0,).normalize().rot()

    mesh.points[:, :] = dot(mesh.points, M.T)
    mesh.points[:, 0] -= 0.2
    mesh.points[:, 1] += 0.25
    mesh.points[:, 2] += 1.9

    #line = array([-0.1,-0.1, 1.0])
    #line = array([-0.43, -0.21, 1.0])
    #line = array([-0.43, -0.21, 1.0])
    line = array([50 - 125.5, 50 - 93, 750.0/4]) / 100

    cm = flattnr.mesh.CameraModel(array([750, 1000])/4, array([500,375.0])/4, 750.0/4)
    pp = cm.project(mesh.points)
    ww = mesh.label_pixels(cm)


    #uv = mgrid[1.5:4.0:0.5,1.5:4.0:0.5].T.reshape(-1,2)
    uv = mgrid[-0.5:6.5:0.01,-0.1:8.2:0.1].T.reshape(-1,2)
    xyz = array([mesh.uv_to_xyz(x) for x in uv])
    pq = array([mesh.uv_to_pq(cm, x) for x in uv])




    Xpq = mgrid[25:175:25,25:175:25].T.reshape(-1,2)
    Xuv = array([mesh.pq_to_uv(cm, pq) for pq in Xpq])
    Xxyz = array([mesh.uv_to_xyz(uv) for uv in Xuv])

    fig = pylab.figure(figsize=(9,9))
    ax = mesh.plot_wireframe()
    ax.set_title('Book page mesh model')
    lx, ly, lz = c_[array([0.0,0.0,0.0]), 2.0 * line]
    ax.plot(lx,lz,ly, 'r-')
    ax.plot([0],[0],[0], 'ko')
    # ax.plot(xyz[:,0], xyz[:,2], xyz[:,1], 'k,')
    for line in Xxyz:
        lx, ly, lz = c_[array([0.0,0.0,0.0]), 1.2 * line]
        ax.plot(lx,lz,ly, 'r-', alpha=0.5)
    ax.plot(Xxyz[:,0], Xxyz[:,2], Xxyz[:,1], 'ro')


    pylab.figure()
    ax = pylab.subplot(1,1,1)    
    ax.set_title('Projected mesh, and pixel labels')
    pylab.imshow(ww)
    plot_2d_mesh(ax, pp, mesh_lins, mesh_cols)
    pylab.axis([0,cm.image_shape[1], cm.image_shape[0],0])
    ax.grid()

    fig = pylab.figure()
    ax = pylab.subplot(1,1,1)    
    ax.set_title('Book page mesh model projection')
    plot_2d_mesh(ax, pp, mesh_lins, mesh_cols)
    ax.axis('equal')
    pylab.axis([0,cm.image_shape[1], cm.image_shape[0],0])
    # pylab.plot(pq[:,0], pq[:,1], 'k,')
    ax.plot(Xpq[:,0], Xpq[:,1], 'rx')
    plot_2d_mesh(ax, Xpq, 6, 6, 'r')
    ax.grid()

    fig = pylab.figure()
    ax = pylab.subplot(1,1,1)    
    ax.set_title('Book page mesh model projection')
    ax.plot(Xuv[:,0], Xuv[:,1], 'rx')

    plot_2d_mesh(ax, Xuv, 6, 6, 'r')

    ax.axis('equal')
    ax.grid()


    import ipdb; ipdb.set_trace();



if __name__ == '__main__':
    main()

