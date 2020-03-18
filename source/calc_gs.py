#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import netket,time
from honeycomb import *
def rbm_gs(size):
    lr=0.01
    n_iter=4000
    hamiltonian=get_hamiltonian(size)
    output_prefix="rbm%d%d_a2_1kx10k"%(size[0],size[1])
    print(output_prefix)
    ma=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,alpha=2)
    ma.init_random_parameters(sigma=0.1)
    t=rbm(hamiltonian,ma,output_prefix,learning_rate=lr,n_iter=n_iter)
    log("%s %.4fs"%(output_prefix,t),l=1)

def rbm_change_lr_gs(size,batch1,batch2,lr1,lr2):
    salt=str(int(time.time())%1000000)
    output_prefix="rbm%d%d_a2_"%(size[0],size[1])+"%dkat%.2f&%dkat%.2f_"%(batch1/1000,lr1,batch2/1000,lr2)+salt
    
    log("going to rbm %s"%(output_prefix))
    hamiltonian=get_hamiltonian(size)
    ma=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,alpha=2)
    ma.init_random_parameters(sigma=0.01)
    t1=rbm(hamiltonian,ma,output_prefix+".1",learning_rate=lr1,n_iter=batch1)
    log("%s %.4fs"%(output_prefix,t1),l=1)
    t2=rbm(hamiltonian,ma,output_prefix+".2",learning_rate=lr2,n_iter=batch2)
    log("%s %.4fs"%(output_prefix,t2),l=1)

if __name__=="__main__":
    log("begin rbm gs",l=1)
    size=(5,5)
    rbm_change_lr_gs(size,5000,5000,0.03,0.01)
    rbm_change_lr_gs(size,4000,6000,0.03,0.01)
    rbm_change_lr_gs(size,3000,7000,0.03,0.01)
    rbm_change_lr_gs(size,2000,8000,0.03,0.01)
    rbm_change_lr_gs(size,1000,9000,0.03,0.01)