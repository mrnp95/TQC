#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import netket,numpy,time,sys
from honeycomb import *
def play_alpha():
    size=(5,5)
    hamiltonian=get_hamiltonian(size)
    for alpha in (0.1,0.3,0.5,1,1.5,2,3,4,6,8,10):
        nh=int(alpha*size[0]*size[1]*2)
        if alpha%1==0:
            output_prefix="rbm%d%d_a%d_1kx10k"%(size[0],size[1],alpha)
        else:
            output_prefix="rbm%d%d_a%.1f_1kx10k"%(size[0],size[1],alpha)
        print(alpha,nh,output_prefix)
        ma=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,n_hidden=nh)
        ma.init_random_parameters(sigma=0.01)
        t=rbm(hamiltonian,ma,"rbm55_a%.1f_1kx10k"%(alpha))
        with open("play_alpha.log","a") as f:
            f.write("%s n_hidden=%d takes %.4fs\n"%(time.strftime("[%Y-%m-%d %H:%M:%S]",time.localtime()),nh,t))

if __name__=="__main__":
    play_alpha()