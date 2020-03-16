#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import netket,time,random
from honeycomb import *

def gen_plaquette(dict_ps,hilbert):
    dict_obs={}
    for k in dict_ps:
        dict_obs[k]=netket.operator.LocalOperator(hilbert,[sx,sy,sz,sx,sy,sz],[[i] for i in dict_ps[k]])
        log("operator %s acting on: %s"%(k,dict_obs[k].acting_on))
    return dict_obs
        
def rbm_gs(size):
    alpha=2
    lrs=[0.05,0.02]
    n_iters=[400,1000]
    n_samples=[400,1000]
    log("lrs: %s"%(lrs))
    log("n_iters: %s"%(n_iters))
    log("n_samples: %s"%(n_samples))
    output_prefix="%dx%d_a%d.%d"%(size[0],size[1],alpha,random.randint(1000,2000))
    log(output_prefix)

    hamiltonian=get_hamiltonian(size)
    dict_focuson={"A":[8,9,16,15,14,7],'B':[2,3,8,7,6,1],'C':[4,5,10,9,8,3],'D':[10,11,12,17,16,9]}
    dict_obs=gen_plaquette(dict_focuson,hamiltonian.hilbert)
    ma=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,alpha=alpha)
    ma.init_random_parameters(seed=1234,sigma=0.1)

    for stage in range(len(lrs)):
        sa=netket.sampler.MetropolisLocal(machine=ma)
        op=netket.optimizer.Sgd(learning_rate=lrs[stage])
        gs=netket.variational.Vmc(hamiltonian=hamiltonian,sampler=sa,optimizer=op,n_samples=n_samples[stage]
                                  ,diag_shift=0.1,use_iterative=True,method='Sr')
        for k in dict_obs:
            gs.add_observable(dict_obs[k],k)

        start=time.time()
        gs.run(output_prefix=output_prefix+".%d"%(stage),n_iter=n_iters[stage])
        end=time.time()
        log('optimize stage %d takes %fs'%(stage,end-start,))
    log("finish")


if __name__=="__main__":
    size=(3,3)
    rbm_gs(size)