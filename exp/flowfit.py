#!/usr/bin/python2.7
#coding: utf-8


from pylab import *


class Snake:

    def __init__(self, N):
        self.state = zeros((N, 2))

    


def vector_field(x):
    v = zeros(x.shape)

    v[:,0] = -x[:,1]
    v[:,1] = x[:,0]
    
    nf = c_[2*[sqrt(v[:,0]**2 + v[:,1]**2)]].T

    print nf

    v = v / nf

    return v
    

if __name__ == '__main__':
    
    ss = Snake(10)


    xx = rand(100,2)

    vv = vector_field(xx)/20.0

    ion()
    plot(xx[:,0], xx[:,1], '.')
    plot(xx[:,0] + c_[-vv[:,0], vv[:,0]].T,
         xx[:,1] + c_[-vv[:,1], vv[:,1]].T,
         'r-')
    axis('equal')


    
    
