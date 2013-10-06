#!/usr/bin/python
#coding:utf-8

import code 

import flattnr.mesh

import argparse

from numpy import array, reshape, c_, mgrid, pi, sin, cos, dot
from numpy.linalg import det, norm

import pylab

from flattnr.quaternion import Quat

def plot_2d_mesh(ax, pp, Nrows, Ncols, color='b', alpha=1.0):
    mesh = pp.reshape(Nrows, Ncols, -1)
    for pp in mesh:
        ax.plot(pp[:,0], pp[:,1], '-', color=color, alpha=alpha)
    for pp in [mesh[:,k,:] for k in xrange(mesh.shape[1])]:
        ax.plot(pp[:,0], pp[:,1], '-', color=color, alpha=alpha)
    pp = reshape(mesh, (-1,2))
    ax.plot(pp[:,0], pp[:,1], 'o', color=color)

def test_label():
    pass

def setup_mesh(rotation, translation, curvature, scale=0.2):
    mesh_rows = 9
    mesh_cols = 7
    mesh = flattnr.mesh.Mesh(mesh_rows, mesh_cols)

    # angle = (pi * 2 / (mesh_cols-1)) / 5
    tt = (mgrid[0.0 : mesh_cols] - mesh_cols / 2) / (mesh_cols-1) * (pi * 2 * curvature)
    circ = c_[sin(tt), 1 - cos(tt)] * (mesh_cols - 1) / (pi * 2 * curvature)

    for ii in range(mesh.Nrows):
        mesh.mesh[ii, :, 0] = circ[:,0] * scale
        mesh.mesh[ii, :, 2] = circ[:,1] * scale
        mesh.mesh[ii, :, 1] = (ii - mesh.Nrows / 2) * scale

    M = rotation.normalize().rot()
    mesh.points[:, :] = dot(mesh.points, M.T)
    mesh.points += translation

    return mesh

def main():
    ## Command-line argument parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--reverse', default=False, action='store_true')

    args = parser.parse_args()

    cm_image_shape = array([750, 1000])/4
    cm_optical_center = array([500, 375.0])/4
    cm_focal_distance = 750.0/4
    cm = flattnr.mesh.CameraModel(cm_image_shape, cm_optical_center, cm_focal_distance)

    Q1 = Quat(1.0, 0.08, -0.1, -0.1)
    T1 = array([-0.2, 0.0, 1.9])
    mesh = setup_mesh(Q1, T1, 0.25)
    mesh.calculate_derivative()

    Q2 = Quat(1.0, 0.09, -0.11, -0.09)
    T2 = array([-0.2, 0.0, 1.9])
    nesh = setup_mesh(Q2, T2, 0.05)
    nesh.calculate_derivative()



    mesh_image = cm.project(mesh.points)
    nesh_image = cm.project(nesh.points)

    pq = mgrid[35:175:10,5:195:10].T.reshape(-1,2)
    uv = array([mesh.pq_to_uv(cm, ipq) for ipq in pq])
    sel = (uv[:,0] > 0) * (uv[:,0] < 6) * (uv[:,1] > 0) * (uv[:,1] < 8)
    pq = pq[sel]
    uv = uv[sel]
    xyz = array([mesh.uv_to_xyz(iuv) for iuv in uv])

    uv2 = array([nesh.pq_to_uv(cm, ipq) for ipq in pq])
    xyz2 = array([nesh.uv_to_xyz(iuv) for iuv in uv2])





    fig = pylab.figure(figsize=(9,9))
    ax = mesh.plot_wireframe()
    ax.set_title('Book page mesh model - 3D')

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
    plot_2d_mesh(ax, mesh_image, mesh.Nrows, mesh.Ncols, color='b', alpha=0.4)
    plot_2d_mesh(ax, nesh_image, mesh.Nrows, mesh.Ncols, color='r', alpha=0.4)
    ax.axis('equal')
    pylab.axis([0,cm.image_shape[1], cm.image_shape[0],0])

    pylab.plot(pq[:,0], pq[:,1], 'k.')

    J = cm.jacobian_from_pq(pq)

    for iuv, ixyz, ipq, iJ in zip(uv, xyz, pq, J):
        dd = dot(iJ, mesh.uv_to_xyz_dev(iuv))
        dd = dd / norm(dd)

        # dd = ipq + 5 * dd
        pylab.plot([ipq[0]-5*dd[0], ipq[0]+5*dd[0]],
                   [ipq[1]-5*dd[1], ipq[1]+5*dd[1]], 'b-', lw=2)
        # pylab.plot([ipq[0], dd[0]],
        #            [ipq[1], dd[1]], 'b-', lw=1.5)

    for iuv, ixyz, ipq, iJ in zip(uv2, xyz2, pq, J):
        dd = dot(iJ, nesh.uv_to_xyz_dev(iuv))
        dd = dd / norm(dd)

        # dd = ipq + 5 * dd
        pylab.plot([ipq[0]-5*dd[0], ipq[0]+5*dd[0]],
                   [ipq[1]-5*dd[1], ipq[1]+5*dd[1]], 'r-', lw=2)
        # pylab.plot([ipq[0], dd[0]],
        #            [ipq[1], dd[1]], 'r-', lw=1.5)

    ax.grid()



    import ipdb; ipdb.set_trace();



if __name__ == '__main__':
    main()

