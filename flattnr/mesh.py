
from numpy import array, dot, zeros, ones, reshape, c_
from numpy.linalg import norm

from scipy.spatial import KDTree
from scipy.optimize import leastsq
from scipy.ndimage.filters import convolve

ando3 = array([[0.112737, 0, -0.112737],
               [0.274526, 0, -0.274526],
               [0.112737, 0, -0.112737]])

get_coefs = array([
        [ 6.0,-12.0, 6.0, 6.0,-12.0, 6.0, 6.0,-12.0, 6.0],
        [ 6.0, 6.0, 6.0,-12.0,-12.0,-12.0, 6.0, 6.0, 6.0],
        [ 9.0,-0.0,-9.0, 0.0,-0.0, 0.0,-9.0,-0.0, 9.0],
        [-6.0, 0.0, 6.0,-6.0, 0.0, 6.0,-6.0, 0.0, 6.0],
        [-6.0,-6.0,-6.0,-0.0, 0.0,-0.0, 6.0, 6.0, 6.0],
        [-4.0, 8.0,-4.0, 8.0,20.0, 8.0,-4.0, 8.0,-4.0]
        ]) / 36.0


class CameraModel(object):
    def __init__(self, image_shape, cp, fd):
        self.image_shape = image_shape
        self.cp = cp
        self.fd = fd

    def project(self, p):
        return self.cp + self.fd * (p[:, [0, 1]] / p[:,[2, 2]])

    def ray(self, pq):
        return c_[(pq - self.cp) / self.fd, ones(len(pq))]

    def jacobian_from_pq(self, pq):
        '''Projection Jacobian from a simple pinhole model.'''
        X, Y = (pq - self.cp) / self.fd
        Z = 1.0

        J = array([[Z, 0, -X],
                   [0, Z, -Y]])
        return J

    def draw_camera(self, ax):
        pq = array([
                [0,0],
                [self.image_shape[1], 0],
                [self.image_shape[1], self.image_shape[0]],
                [0, self.image_shape[0]],
                ])
        cam_xyz = self.ray(pq) * 0.5
        for ii in range(4):
            ax.plot([cam_xyz[ii, 0], cam_xyz[(ii + 1) % 4, 0]],
                    [cam_xyz[ii, 2], cam_xyz[(ii + 1) % 4, 2]],
                    [cam_xyz[ii, 1], cam_xyz[(ii + 1) % 4, 1]],
                    'r-')
            ax.plot([0, cam_xyz[ii, 0]],
                    [0, cam_xyz[ii, 2]],
                    [0, cam_xyz[ii, 1]],
                    'r-')

class Mesh(object):
    def __init__(self, Nrows, Ncols):
        self.Nrows = Nrows
        self.Ncols = Ncols
        self.points = zeros((self.Nrows * self.Ncols, 3))
        self.mesh = self.points.reshape(self.Nrows, self.Ncols, 3)

    def plot_wireframe(self):
        import pylab
        from mpl_toolkits.mplot3d import Axes3D

        pylab.ion()
        fig = pylab.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        # x=2*np.cos(u)*np.sin(v)
        # y=np.sin(u)*np.sin(v)
        # z=np.cos(v)
        # ax.plot_wireframe(x, y, z, color="r")
        # ax.set_aspect('equal')

        ax.plot_wireframe(self.mesh[:,:,0], 
                          self.mesh[:,:,2], 
                          self.mesh[:,:,1])

        angle = (pylab.pi * 2 / (self.Ncols - 1)) / 5
        # ll = (self.points[:,:,1].max() - self.points[:,:,1].min())/2
        ll = 1
        ax.set_xlim(-ll,ll)
        ax.set_zlim(ll,-ll)
        #ax.set_ylim(1-ll,1+ll)
        ax.set_ylim(0,2*ll)

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        return ax

    def label_pixels(self, camera):
        img_lab = zeros(camera.image_shape)

        pp = camera.project(self.points)
        tree = KDTree(pp)

        for il in xrange(int(camera.image_shape[0])):
            for ic in xrange(int(camera.image_shape[1])):
                pixel = array([ic, il])
                img_lab[il, ic] = tree.query(pixel)[1]
        return img_lab

    def uv_to_xyz(self, uv):
        mesh_row = round(uv[1])
        mesh_col = round(uv[0])

        if mesh_row < 1:
            mesh_row = 1
        if mesh_col < 1:
            mesh_col = 1
        if mesh_row > self.Nrows - 2:
            mesh_row = self.Nrows - 2
        if mesh_col > self.Ncols - 2:
            mesh_col = self.Ncols - 2

        lx, ly = uv - (mesh_col, mesh_row)

        out = zeros(3)
        for c in range(3):
            local_mesh = self.mesh[mesh_row - 1 : mesh_row + 2,
                                   mesh_col - 1 : mesh_col + 2, c]

            coefs = dot(get_coefs, local_mesh.ravel())
        
            out[c] = dot(coefs, [lx * lx, ly * ly, lx * ly, lx, ly, 1])
        return out

    def uv_to_pq(self, cm, uv):
        return cm.project(self.uv_to_xyz(uv))

    def pq_to_uv(self, cm, pq):
        def err(xx, cm):
            vv = self.uv_to_pq(cm, xx) - pq
            return vv * vv

        return leastsq(err, array([3.0, 4.0]), args=(cm,))[0]

    def uv_to_xyz_dev(self, uv):
        mesh_row = round(uv[1])
        mesh_col = round(uv[0])

        if mesh_row < 1:
            mesh_row = 1
        if mesh_col < 1:
            mesh_col = 1
        if mesh_row > self.Nrows - 2:
            mesh_row = self.Nrows - 2
        if mesh_col > self.Ncols - 2:
            mesh_col = self.Ncols - 2

        lx, ly = uv - (mesh_col, mesh_row)

        out = zeros(3)
        for c in range(3):
            local_mesh = self.mesh_dev[mesh_row - 1 : mesh_row + 2,
                                       mesh_col - 1 : mesh_col + 2, c]
            coefs = dot(get_coefs, local_mesh.ravel())
            out[c] = dot(coefs, [lx * lx, ly * ly, lx * ly, lx, ly, 1])
        return out

    def calculate_derivative(self):
        self.mesh_dev = zeros([self.Nrows, self.Ncols, 3])
        self.mesh_devy = zeros([self.Nrows, self.Ncols, 3])

        for cc in range(3):
            self.mesh_dev[:,:,cc] = convolve(self.mesh[:,:,cc], ando3, mode='nearest')
            self.mesh_devy[:,:,cc] = convolve(self.mesh[:,:,cc], ando3.T, mode='nearest')

    def calculate_stress(self):
        
        stress = 0.0
        for u in range(1,self.Ncols - 1):
            for v in range(1,self.Nrows - 1):
            
                stress += (norm(self.mesh_dev[v, u]) - 0.2)**2
                stress += (norm(self.mesh_devy[v, u]) - 0.2)**2

        return stress
