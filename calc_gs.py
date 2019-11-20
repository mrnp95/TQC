#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import netket
from honeycomb import *
def rbm_gs(size):
    hamiltonian=get_hamiltonian(size)
    output_prefix="rbm%d%d_a2_1kx10k"%(size[0],size[1])
    print(output_prefix)
    ma=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,alpha=2)
    ma.init_random_parameters(sigma=0.01)
    t=rbm(hamiltonian,ma,output_prefix)
    log("%s %.4fs"%(output_prefix,t),l=1)

if __name__=="__main__":
    log("begin rbm gs",l=1)
    for size in ((3,3),(4,4)):
        rbm_gs(size)
