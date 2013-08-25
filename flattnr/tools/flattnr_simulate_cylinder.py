#!/usr/bin/python
#coding:utf-8

import code 

import flattnr.mesh

import argparse

from numpy import array, reshape, c_, mgrid, pi, sin, cos, dot
from numpy.linalg import det

import pylab

from flattnr.quaternion import Quat

def plot_2d_mesh(ax, pp, Nlins, Ncols):
    mesh = pp.reshape(Nlins, Ncols, -1)
    for pp in mesh:
        ax.plot(pp[:,0], pp[:,1], 'b-')
    for pp in [mesh[:,k,:] for k in xrange(mesh.shape[1])]:
        ax.plot(pp[:,0], pp[:,1], 'b-')
    pp = reshape(mesh, (-1,2))
    ax.plot(pp[:,0], pp[:,1], 'bo')

def main():
    ## Command-line argument parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_file', type=str)

    args = parser.parse_args()

    mesh_lins = 9
    mesh_cols = 7
    mesh = flattnr.mesh.Mesh(mesh_lins, mesh_cols)

    angle = (pi * 2 / (mesh_cols-1)) / 5
    tt = angle * (mgrid[0.0:mesh_cols] - mesh_cols / 2)
    circ = c_[sin(tt), -cos(tt)]

    for ii in range(mesh_lins):
        mesh.mesh[ii, :, 0] = circ[:,0]
        mesh.mesh[ii, :, 2] = circ[:,1] + 1
        mesh.mesh[ii, :, 1] = (mesh_cols/2 - ii) * angle

    M = Quat(1.0,0.08,-0.1,-0.1).normalize().rot()

    mesh.points[:, :] = dot(mesh.points, M.T)
    mesh.points[:, 0] -= 0.2
    mesh.points[:, 1] += 0.25
    mesh.points[:, 2] += 1.9

    tri = [(0,0), (0,1), (1,0)]
    #line = array([-0.1,-0.1, 1.0])
    line = array([-0.43, -0.21, 1.0])
    mesh.line_crosses_triangle(line, tri)






    fig = pylab.figure(figsize=(9,9))
    ax = mesh.plot_wireframe()
    ax.set_title('Book page mesh model')
    lx, ly, lz = c_[array([0.0,0.0,0.0]), 2.0 * line]
    ax.plot(lx,lz,ly, 'r-')
    ax.plot([0],[0],[0], 'ko')
    # code.interact(local=locals)



    




    cm = flattnr.mesh.CameraModel(array([750, 1000])/4, array([500,375.0])/4, 750.0/4)

    pp = cm.project(mesh.points)

    
    fig = pylab.figure()
    ax = pylab.subplot(1,1,1)    
    ax.set_title('Book page mesh model projection')
    plot_2d_mesh(ax, pp, mesh_lins, mesh_cols)
    ax.axis('equal')
    pylab.axis([0,cm.image_shape[1], cm.image_shape[0],0])
    ax.grid()



    # pylab.plot(pp[:,0], pp[:,1], 'bo')

    ww = mesh.render_image(cm)
    print ww

    pylab.figure()
    ax = pylab.subplot(1,1,1)    
    ax.set_title('Projected mesh, and pixel labels')
    pylab.imshow(ww)
    plot_2d_mesh(ax, pp, mesh_lins, mesh_cols)
    pylab.axis([0,cm.image_shape[1], cm.image_shape[0],0])
    ax.grid()


    import ipdb; ipdb.set_trace();




if __name__ == '__main__':
    main()

