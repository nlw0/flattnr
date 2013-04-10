#!/usr/bin/python2.7
#coding: utf-8


from pylab import *

import itertools

class Snake:

    def __init__(self, N):
        self.N = N
        self.state = zeros((self.N, 2))
        #self.state[0] = (0.0, 5.0)
        #self.state = random((self.N, 2))
        self.x0 = array([0.0, 5.0])
        self.state[:,0] = mgrid[:self.N]

    def derivative(self):
        return 0.5 * (self.state[:-2] - self.state[2:])

    def derivative2(self):
        return 0.5 * (self.state[:-2] + self.state[2:] - 2 * self.state[1:-1])

    def project(self):
        new_state = copy(self.state)
        new_state -= new_state[0]
        new_state += self.x0
        for n in range(1, self.N):
            d = new_state[n] - new_state[n - 1]
            l = norm(d + 1e-10)
            new_state[n:] -= new_state[n]
            new_state[n:] += new_state[n - 1] + d / l
        self.state = new_state

    def update_from_vf(self, vf):
        dd = self.derivative()
        qq = vf(self.state[1:-1])
        for k in range(self.N-3):
            err = -10.*(dd[k] - qq[k])
            self.state[k] += err
            self.state[k+2] += -err

    def update_by_rigidity(self):
        for k in range(self.N-3):
            v1 = self.state[k + 1] - self.state[k]
            v2 = self.state[k + 2] - self.state[k + 1]
            v2o = array([-v2[1], v2[0]])

            dd = v1[0] * v2o[0] + v1[1] * v2o[1]

            ff = 10.0

            self.state[k+2:] += dd * ff * v2o




    def plot(self, ax, style='k-o'):
        ax.plot(self.state[:,0], self.state[:,1], style)


def vector_field(x):
    v = zeros(x.shape)
    v[:,0] = -x[:,1]
    v[:,1] = x[:,0]
    nf = c_[2*[sqrt(v[:,0]**2 + v[:,1]**2)]].T
    v = v / (nf+1e-10)
    # v[:,0] = 1
    # v[:,1] = 0
    return v


if __name__ == '__main__':

    ss = Snake(10)
    #xx = rand(100,2)

    x1 = 10 * mgrid[:21]/20.0
    xx = array(list(itertools.product(x1, x1)))
    vv = vector_field(xx)/20.0

    ion()
    l = 3.0
    plot(xx[:,0] + l * c_[-vv[:,0], vv[:,0]].T,
         xx[:,1] + l * c_[-vv[:,1], vv[:,1]].T,
         'r-')
    axis('equal')

    ss.plot(gca())
    for k in range(1000):
        ss.update_from_vf(vector_field)
        ss.update_by_rigidity()
        
        ss.project()

    print ss.state
    ss.project()
    ss.plot(gca(), style='g-+')


