#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from honeycomb import *
def check33(flip):
    d_temp={X_FLAG:"x",Y_FLAG:"y",Z_FLAG:"z"}
    output_prefix=["rbm33_"]
    for i in flip:
        output_prefix.append("%d%s"%(i[0],d_temp[i[1]]))
    output_prefix.append("_a2_1kx10k")
    output_prefix="".join(output_prefix)
    log(output_prefix)

    hamiltonian=get_hamiltonian((3,3),flip=flip)
    ma=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,alpha=2)
    ma.init_random_parameters(sigma=0.01)
    t=rbm(hamiltonian,ma,output_prefix)
    log("%s %.4fs"%(output_prefix,t),l=1)

def vortex33(sites):
    logfile=[]
    for i,d in sites:
        logfile.append("%d%s"%(i,d))
    logfile.append("33.measure")
    logfile="".join(logfile)
    log("get logfile name: %s"%(logfile))

    hamiltonian=get_hamiltonian((3,3),show=False)
    obs={}
    for i in range(18):
        obs["sz%d"%(i)]=netket.operator.LocalOperator(hamiltonian.hilbert,[sz,],[[i],])
        obs["sy%d"%(i)]=netket.operator.LocalOperator(hamiltonian.hilbert,[sy,],[[i],])
        obs["sx%d"%(i)]=netket.operator.LocalOperator(hamiltonian.hilbert,[sx,],[[i],])
    machine=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,alpha=2)
    machine.load("./rbm33_a2_1kx10k.wf")
    
    log("begin measuring",l=1,logfile=logfile)
    machine.parameters=flip_spins(machine,36,sites)
    results=measure(hamiltonian,machine,obs,n_iter=1000)
    for k in sorted(results):
        log("%s: %f(%f)"%(k,results[k]["mean_Mean"],results[k]["Error"]),l=1,logfile=logfile)
if __name__=="__main__":
    vortex33([])
    vortex33([(8,"xz"),])
    vortex33([(9,"xz"),])
    check33([(8,0),(8,2)])
    check33([(9,0),(9,2)])
    check33([(8,1),])
    
    