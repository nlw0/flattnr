#!/usr/bin/python
#coding:utf-8

import argparse
import code 

from flattnr.mesh import Mesh, CameraModel
from flattnr.quaternion import Quat

from numpy import array, reshape, c_, mgrid, pi, sin, cos, dot, fromiter
from numpy.linalg import det, norm

import pylab

from scipy.optimize import leastsq

def test_label():
    pass

def extract_edgels(cm, mesh, pq):
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

def main():
    ## Command-line argument parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--reverse', default=False, action='store_true')

    args = parser.parse_args()

    cm_image_shape = array([750, 1000])/4
    cm_optical_center = array([500, 375.0])/4
    cm_focal_distance = 750.0/4
    cm = CameraModel(cm_image_shape, cm_optical_center, cm_focal_distance)

    ## Original parameters and model
    Nrows, Ncols = 9, 7
    Q_ori = Quat(1.0, 0.08, -0.1, -0.1)
    T_ori = array([-0.2, 0.0, 1.9])    
    mesh_ori = Mesh(9, 7, Q_ori, T_ori, 0.25)
    mesh_ori.calculate_derivative()

    ## Analyze a grid of points sampled regularly.
    pq_ini = mgrid[35:175:10,5:195:10].T.reshape(-1,2)

    data = list(extract_edgels(cm, mesh_ori, pq_ini))
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
    curvature = -0.06
    # Q2 = Quat(1.0, 0.08, -0.1, -0.1)
    # T2 = array([-0.2, 0.0, 1.9])
    # curvature = -0.2
    nesh = Mesh(Nrows, Ncols, Q2, T2, curvature)
    nesh.calculate_derivative()
    nesh_image = cm.project(nesh.points)

    x = nesh.points.ravel()

    error = target_function_value(x, Nrows, Ncols, ds_obs, cm, data)

    print "***"
    print error
    print "***"

    print list(target_function(x, Nrows, Ncols, ds_obs, cm, data))
    
    # import ipdb; ipdb.set_trace();

    qq = leastsq(target_function_list, x, args=(Nrows, Ncols, ds_obs, cm, data))[0]
    
    
    mesh_est = Mesh(Nrows, Ncols)
    mesh_est.points[:] = qq.reshape(*mesh_est.points.shape)
    mesh_est.calculate_derivative()

    pylab.ion()

    fig = pylab.figure()
    ax = pylab.subplot(1,1,1)    
    ax.set_title('Book page mesh model projection')

    mesh_ori.plot_2d_mesh(ax, cm, color='b', alpha=0.4, do_vertices=False)
    nesh.plot_2d_mesh(ax, cm, color='r', alpha=0.4, do_vertices=False)
    mesh_est.plot_2d_mesh(ax, cm, color='b', alpha=0.4, do_vertices=False)

    ds_est = fromiter((x for p in get_edgel_directions(cm, nesh, data)
                       for x in p), dtype=float, count=(Npoints * 2))
    ds_est.resize(Npoints, 2)

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

def target_function_value(x, Nrows, Ncols, ds_obs, cm, edgel_locations):
    return sum(err*err for err in target_function(x, Nrows, Ncols, ds_obs, cm, edgel_locations))

def target_function_list(x, Nrows, Ncols, ds_obs, cm, edgel_locations):
    return [f for f in target_function(x, Nrows, Ncols, ds_obs, cm, edgel_locations)]

def target_function(x, Nrows, Ncols, ds_obs, cm, edgel_locations):
    ## Create a new mesh model with "incorrect" parameters.
    mesh = Mesh(Nrows, Ncols)
    mesh.points[:] = x.reshape(*mesh.points.shape)
    mesh.calculate_derivative()

    error = 0.0
    for d_est, d_obs in zip(get_edgel_directions(cm, mesh, edgel_locations), ds_obs):
        yield d_est[0] * d_obs[1] - d_est[1] * d_obs[0]

    for st in mesh.stress_terms():
        yield st - 0.2

def get_edgel_directions(cm, mesh, edgel_locations):
    for pp in edgel_locations:
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

if __name__ == '__main__':
    main()

