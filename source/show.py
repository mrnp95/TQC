#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys,json,numpy,os,time
import matplotlib.pyplot as plt
from honeycomb import log

def show_with_plaquette(filename,exact_gs_energy=None):
    energy={"Mean":[],"Sigma":[]}
    obs={"A":{"Mean":[]},"B":{"Mean":[]},"C":{"Mean":[]},"D":{"Mean":[]},}

    log(1)
    with open(filename+"0.log") as f:
        data=json.load(f)["Output"]
    for stage in range(1,2):
        if os.path.exists(filename+"%d.log"%(stage)):
            with open(filename+"%d.log"%(stage)) as f:
                data+=json.load(f)["Output"]
    log(2)
    for i in data:
        energy["Mean"].append(i["Energy"]["Mean"])
        energy["Sigma"].append(i["Energy"]["Sigma"])
        for k in obs:
            obs[k]["Mean"].append(i[k]["Mean"])
    log(3)
    iter_num=len(energy["Mean"])
    smooth_num=5
    energy["Mean"]=[numpy.mean(energy["Mean"][max(i-smooth_num,0):min(i+smooth_num,iter_num)]) for i in range(iter_num)]
    smooth_num_2=10
    for k in obs:
        obs[k]["Mean"]=[numpy.mean(obs[k]["Mean"][max(i-smooth_num_2,0):min(i+smooth_num_2,iter_num)]) for i in range(iter_num)]
    log(4)
    iters=list(range(1,iter_num+1))
    fig,ax1=plt.subplots(1)
    ax1.plot(iters,energy["Mean"],c='navy',label='RBM Energy')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    if exact_gs_energy!=None:
        plt.axhline(y=exact_gs_energy,c="green",xmin=0,xmax=1,label='Exact=%.4f'%(exact_gs_energy))
        delta_energy=numpy.abs(exact_gs_energy-energy["Mean"][-1])
        ax1.axis([0,iter_num,exact_gs_energy-0.1*delta_energy,exact_gs_energy+3*delta_energy])
    ax1.legend(loc=3)
    
    ax2=ax1.twinx()
    for k in obs:
        ax2.plot(iters,obs[k]["Mean"],label="Plaquette "+k)
    ax2.set_ylabel('Plaquette Operator')
    ax2.axis([0,iter_num,-1,2])
    ax2.legend(loc=1)
    log(5)
    pic_name=filename.split(".")
    pic_name[-1]="png"
    plt.savefig(".".join(pic_name))
    log(6)
    plt.show()

if __name__=="__main__":
    show_with_plaquette(sys.argv[1],exact_gs_energy=-14.2915)