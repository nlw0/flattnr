#!/usr/bin/python
#coding:utf-8

import code 

import flattnr.mesh

import argparse

from numpy import array, reshape, c_, mgrid, pi, sin, cos, dot, fromiter
from numpy.linalg import det, norm

import pylab

from flattnr.quaternion import Quat

def plot_2d_mesh(ax, pp, Nrows, Ncols, color='b', alpha=1.0, do_vertices=True):
    mesh = pp.reshape(Nrows, Ncols, -1)
    for pp in mesh:
        ax.plot(pp[:,0], pp[:,1], '-', color=color, alpha=alpha)
    for pp in [mesh[:,k,:] for k in xrange(mesh.shape[1])]:
        ax.plot(pp[:,0], pp[:,1], '-', color=color, alpha=alpha)
    if do_vertices:
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


def get_points_to_analyze(cm, mesh, pq):
    for ipq in pq:
        uv = mesh.pq_to_uv(cm, ipq)
        sel = (uv[0] > 0) * (uv[0] < 6) * (uv[1] > 0) * (uv[1] < 8)
        if not sel:
            continue

        iJ = cm.jacobian_from_pq(ipq)
        yield {
            "pq" : ipq,
            "J" : iJ,
        }    

def get_edgel_directions(cm, mesh, points):
    for pp in points:
        ipq, iJ = pp['pq'], pp['J']

        uv = mesh.pq_to_uv(cm, ipq)        

        ## Calculate text direction on model space. Should have
        ## unit norm imposed by model restrictions.
        dr = mesh.uv_to_xyz_dev(uv)

        # Calculate text direction in image space. Normalization
        # is necessary because of distortion from projection.
        ds = dot(iJ, dr)
        ds = ds / norm(ds)
        yield ds


def main():
    ## Command-line argument parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--reverse', default=False, action='store_true')

    args = parser.parse_args()

    cm_image_shape = array([750, 1000])/4
    cm_optical_center = array([500, 375.0])/4
    cm_focal_distance = 750.0/4
    cm = flattnr.mesh.CameraModel(cm_image_shape, cm_optical_center, cm_focal_distance)

    ## Original parameters and model
    Q_ori = Quat(1.0, 0.08, -0.1, -0.1)
    T_ori = array([-0.2, 0.0, 1.9])    
    mesh_ori = setup_mesh(Q_ori, T_ori, 0.25)
    mesh_ori.calculate_derivative()

    ## Analyze a grid of points sampled regularly.
    pq_ini = mgrid[35:175:10,5:195:10].T.reshape(-1,2)

    data = list(get_points_to_analyze(cm, mesh_ori, pq_ini))
    Npoints = len(data)

    pq = fromiter((x for p in data for x in p['pq']), dtype=float, count=(Npoints * 2))
    pq.resize(Npoints, 2)
    J = fromiter((x for p in data for x in p['J'].ravel()), dtype=float, count=(Npoints * 6))
    J.resize(Npoints, 2, 3)

    ## Measured edgel directions
    ds_obs = fromiter((x for p in get_edgel_directions(cm, mesh_ori, data)
                       for x in p), dtype=float, count=(Npoints * 2))
    ds_obs.resize(Npoints, 2)

    ## Create a new mesh model with "incorrect" parameters.
    Q2 = Quat(1.0, 0.02, -0.09, -0.04)
    T2 = array([-0.2, 0.0, 1.9])
    nesh = setup_mesh(Q2, T2, -0.06)
    nesh.calculate_derivative()
    nesh_image = cm.project(nesh.points)

    ds_est = fromiter((x for p in get_edgel_directions(cm, nesh, data)
                       for x in p), dtype=float, count=(Npoints * 2))
    ds_est.resize(Npoints, 2)


    error = 0.0
    for ipq, ids, idr in zip(pq, ds_obs, ds_est):
        ierr = ids[0] * idr[1] + ids[1] * idr[0]
        ierr = ierr * ierr
        error += ierr
        

    print "***"
    print error
    print mesh_ori.calculate_stress()
    print nesh.calculate_stress()

    mesh_image = cm.project(mesh_ori.points)
    nesh_image = cm.project(nesh.points)

    pylab.ion()

    fig = pylab.figure()
    ax = pylab.subplot(1,1,1)    
    ax.set_title('Book page mesh model projection')

    plot_2d_mesh(ax, mesh_image, mesh_ori.Nrows, mesh_ori.Ncols, color='b', alpha=0.4, do_vertices=False)
    plot_2d_mesh(ax, nesh_image, mesh_ori.Nrows, mesh_ori.Ncols, color='r', alpha=0.4, do_vertices=False)

    for ipq, ids, idr in zip(pq, ds_obs, ds_est):
        pylab.plot([ipq[0]-5*ids[0], ipq[0]+5*ids[0]],
                   [ipq[1]-5*ids[1], ipq[1]+5*ids[1]], 'b-', lw=2)
        pylab.plot([ipq[0]-5*idr[0], ipq[0]+5*idr[0]],
                   [ipq[1]-5*idr[1], ipq[1]+5*idr[1]], 'r-', lw=2)
    pylab.plot(pq[:,0], pq[:,1], 'k.')

    ax.axis('equal')
    pylab.axis([0,cm.image_shape[1], cm.image_shape[0],0])
    ax.grid()



    import ipdb; ipdb.set_trace();



if __name__ == '__main__':
    main()

