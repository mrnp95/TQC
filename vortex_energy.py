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
def vortex_energy(size,sites):
    #below 81 is 98, above 80 is 61 for 9x9 lattice
    logfile=[]
    for i,d in sites:
        logfile.append("%d%s"%(i,d))
    logfile.append("%d%d.measure"%(size))
    logfile="".join(logfile)
    log("get logfile name: %s"%(logfile))

    hamiltonian=get_hamiltonian(size,show=False)
    machine=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,alpha=2)
    machine.load("./rbm77_a2_1kx10k.wf")
    
    log("begin measuring",l=1,logfile=logfile)
    machine.parameters=flip_spins(machine,size[0]*size[1]*2*2,sites)
    try:
        results=measure(hamiltonian,machine,{},n_iter=100,n_samples=100)
        for k in sorted(results):
            log("%s: %f(%f)"%(k,results[k]["mean_Mean"],results[k]["Error"]),l=1,logfile=logfile)
    except:
        log(l=3,logfile=logfile)
if __name__=="__main__":
    #vortex_energy((7,7),[(49,"xy"),])
    vortex_energy((7,7),[(49,"xy"),(51,"xy")])
    vortex_energy((7,7),[(47,"xy"),(49,"xy"),(51,"xy")])
    vortex_energy((7,7),[(47,"xy"),(49,"xy"),(51,"xy"),(53,"xy")])

    vortex_energy((7,7),[(47,"xz"),(48,"xy"),(49,"yz")])
    vortex_energy((7,7),[(47,"xz"),(48,"xy"),(49,"yz"),(51,"xz"),(52,"xy"),(53,"yz")])
    

    #vortex99([(81,"xy"),(83,"xy")])
    #vortex99([(81,"xy"),(83,"xy"),(85,"xy")])
    #vortex99([(81,"xy"),(83,"xy"),(85,"xy"),(87,"xy")])
    
    #vortex99([(81,"zy"),(82,"yx"),(83,"xz")])
    #vortex99([(81,"zy"),(82,"yx"),(83,"xz"),(85,"zy"),(86,"yx"),(87,"xz"),])

    #vortex33([])
    #vortex33([(8,"xz"),])
    #vortex33([(9,"xz"),])
    #check33([(8,0),(8,2)])
    #check33([(9,0),(9,2)])
    #check33([(8,1),])
    
    