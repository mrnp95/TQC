#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import netket,numpy,time,sys
from honeycomb import *
def try_ffnn():
    size=(5,5)
    n=2*size[0]*size[1]
    hamiltonian=get_hamiltonian(size)
    for alpha in (1,2,3):
        nh=int(alpha*n)
        layers=(netket.layer.FullyConnected(input_size=n,output_size=nh,use_bias=True),
                netket.layer.Lncosh(input_size=nh),
                netket.layer.SumOutput(input_size=nh))
        for layer in layers:
            layer.init_random_parameters(sigma=0.01)
        ffnn=netket.machine.FFNN(hamiltonian.hilbert,layers)
        t1=rbm(hamiltonian,ffnn,"ffnn55_nh%dx1_1kx10k"%(nh))
        with open("try_ffnn.log","a") as f:
            f.write("%s ffnn with deep=1 nh=%d takes %.4fs\n"%(time.strftime("[%Y-%m-%d %H:%M:%S]",time.localtime()),nh,t1))
        
        ma=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,n_hidden=nh)
        ma.init_random_parameters(sigma=0.01)
        t2=rbm(hamiltonian,ma,"rbm55_nh%d_novb_1kx10k"%(nh))
        with open("try_ffnn.log","a") as f:
            f.write("%s rbm no vb nh=%d takes %.4fs\n"%(time.strftime("[%Y-%m-%d %H:%M:%S]",time.localtime()),nh,t2))
def try_deep_ffnn():
    size=(5,5)
    n=2*size[0]*size[1]
    hamiltonian=get_hamiltonian(size)
    
    nh=n
    layers=(netket.layer.FullyConnected(input_size=n,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.FullyConnected(input_size=nh,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.SumOutput(input_size=nh))
    for layer in layers:
        layer.init_random_parameters(sigma=0.01)
    ffnn=netket.machine.FFNN(hamiltonian.hilbert,layers)
    t1=rbm(hamiltonian,ffnn,"ffnn55_nh%dx2_1kx10k"%(nh))
    with open("try_deep_ffnn.log","a") as f:
        f.write("%s ffnn with deep=2 nh=%d takes %.4fs\n"%(time.strftime("[%Y-%m-%d %H:%M:%S]",time.localtime()),nh,t1))

    layers=(netket.layer.FullyConnected(input_size=n,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.FullyConnected(input_size=nh,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.FullyConnected(input_size=nh,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.SumOutput(input_size=nh))
    for layer in layers:
        layer.init_random_parameters(sigma=0.01)
    ffnn=netket.machine.FFNN(hamiltonian.hilbert,layers)
    t1=rbm(hamiltonian,ffnn,"ffnn55_nh%dx3_1kx10k"%(nh))
    with open("try_deep_ffnn.log","a") as f:
        f.write("%s ffnn with deep=3 nh=%d takes %.4fs\n"%(time.strftime("[%Y-%m-%d %H:%M:%S]",time.localtime()),nh,t1))
    
    nh=int(2*n)
    layers=(netket.layer.FullyConnected(input_size=n,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.FullyConnected(input_size=nh,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.SumOutput(input_size=nh))
    for layer in layers:
        layer.init_random_parameters(sigma=0.01)
    ffnn=netket.machine.FFNN(hamiltonian.hilbert,layers)
    t1=rbm(hamiltonian,ffnn,"ffnn55_nh%dx2_1kx10k"%(nh))
    with open("try_deep_ffnn.log","a") as f:
        f.write("%s ffnn with deep=2 nh=%d takes %.4fs\n"%(time.strftime("[%Y-%m-%d %H:%M:%S]",time.localtime()),nh,t1))

    layers=(netket.layer.FullyConnected(input_size=n,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.FullyConnected(input_size=nh,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.FullyConnected(input_size=nh,output_size=nh,use_bias=True),
            netket.layer.Lncosh(input_size=nh),
            netket.layer.SumOutput(input_size=nh))
    for layer in layers:
        layer.init_random_parameters(sigma=0.01)
    ffnn=netket.machine.FFNN(hamiltonian.hilbert,layers)
    t1=rbm(hamiltonian,ffnn,"ffnn55_nh%dx3_1kx10k"%(nh))
    with open("try_deep_ffnn.log","a") as f:
        f.write("%s ffnn with deep=3 nh=%d takes %.4fs\n"%(time.strftime("[%Y-%m-%d %H:%M:%S]",time.localtime()),nh,t1))
if __name__=="__main__":
    try_ffnn()
    try_deep_ffnn()