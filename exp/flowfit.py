#!/usr/bin/python2.7
#coding: utf-8


from pylab import *

import itertools

class Snake:

    def __init__(self, N, x0):
        self.N = N
        self.state = zeros((self.N, 2))
        self.state[:,0] = 1
        self.x0 = x0


    def integrate(self):
        out = zeros((self.N+1,2))
        out[0] = self.x0
        acc = array([1.0, 0.0])
        for k in range(self.N):
            M = array([[self.state[k,0], self.state[k,1]],
                       [-self.state[k,1], self.state[k,0]]])
            acc = dot(acc, M)
            out[k+1] = out[k] + acc

        return out

    def derivative(self):
        return 0.5 * (self.state[:-2] - self.state[2:])

    def derivative2(self):
        return 0.5 * (self.state[:-2] + self.state[2:] - 2 * self.state[1:-1])

    def project(self):
        self.state = self.state / c_[2*[sqrt((ss.state**2).sum(1))]].T

    def plot(self, ax, style='k-o'):
        xx = self.integrate()
        ax.plot(xx[:,0], xx[:,1], style)


def vector_field(x):
    v = zeros(x.shape)
    # v[:,0] = -x[:,1]
    # v[:,1] = x[:,0]
    # nf = c_[2*[sqrt(v[:,0]**2 + v[:,1]**2)]].T
    # v = v / (nf+1e-10)
    v[:,0] = 1
    v[:,1] = 0
    return v


if __name__ == '__main__':

    ss = Snake(10, array([0.0, 8.0]))

    ss.state += 10.5 * random(ss.state.shape)

    ss.project()


    x1 = 10 * mgrid[:21]/20.0
    xx = array(list(itertools.product(x1, x1)))
    vv = vector_field(xx)/20.0

    ion()
    l = 3.0
    plot(xx[:,0] + l * c_[-vv[:,0], vv[:,0]].T,
         xx[:,1] + l * c_[-vv[:,1], vv[:,1]].T,
         'r-')
    axis('equal')

    ## Initial state
    ss.plot(gca())


    ## Do stuff
    ss.





    ss.plot(gca(), style='g-+')


