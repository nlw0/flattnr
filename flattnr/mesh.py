
from numpy import array, dot, zeros, reshape
from numpy.linalg import norm

from scipy.spatial import KDTree

class CameraModel(object):
    def __init__(self, image_shape, cp, fd):
        self.image_shape = image_shape
        self.cp = cp
        self.fd = fd

    def project(self, p):
        return self.cp + self.fd * (p[:, [0, 1]] / p[:,[2, 2]])

        

class Mesh(object):
    def __init__(self, Nlins, Ncols):
        self.Nlins = Nlins
        self.Ncols = Ncols
        self.points = zeros((self.Nlins * self.Ncols, 3))
        self.mesh = self.points.reshape(self.Nlins, self.Ncols, 3)

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

    def render_image(self, camera):

        img_lab = zeros(camera.image_shape)

        pp = camera.project(self.points)
        tree = KDTree(pp)

        for il in xrange(int(camera.image_shape[0])):
            for ic in xrange(int(camera.image_shape[1])):
                pixel = array([ic, il])
                img_lab[il, ic] = tree.query(pixel)[1]

        

        return img_lab

    def line_crosses_triangle(self, line, triangle):
        pa, pb, pc = [self.points[point_index] for point_index in triangle]
        print pa,pb,pc
