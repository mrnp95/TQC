#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys,json,numpy
def show_data_cui(name,avg_num=None):
    """ show log file of netket with cui
        name: the name of file to show
        avg_num: the number of batches to be averaged in the output
    """
    data=json.load(open(name))
    iters=[]
    energy={"Mean":[],"Sigma":[]}
    for i in data["Output"]:
        iters.append(i["Iteration"])
        energy["Mean"].append(i["Energy"]["Mean"])
        energy["Sigma"].append(i["Energy"]["Sigma"])
    iter_num=len(iters)
    if avg_num==None:
        avg_num=int(iter_num/10)
    
    for i in range(iter_num%avg_num,iter_num,avg_num):
        energy_i=numpy.mean(energy["Mean"][i:i+avg_num-1])
        sigma_energy_i=numpy.mean(energy["Sigma"][i:i+avg_num-1])/numpy.sqrt(avg_num)
        print("[%5d,%5d) %14.10f %14.10f"%(i,i+avg_num,energy_i,sigma_energy_i))
    
def show_data(name,avg_num=None,exact_gs_energy=None,smooth_num=None,title=None):
    import matplotlib.pyplot as plt
    data=json.load(open(name))
    iters=[]
    energy={"Mean":[],"Sigma":[]}
    for i in data["Output"]:
        iters.append(i["Iteration"])
        energy["Mean"].append(i["Energy"]["Mean"])
        energy["Sigma"].append(i["Energy"]["Sigma"])
    iter_num=len(iters)
    if avg_num==None:
        avg_num=int(iter_num/10)
    energy_last=numpy.mean(energy["Mean"][iter_num-avg_num:iter_num])
    sigma_energy_last=numpy.mean(energy["Sigma"][iter_num-avg_num:iter_num])/numpy.sqrt(avg_num)
    
    if smooth_num==None:
        smooth_num=2
    smoothed_energy=[numpy.mean(energy["Mean"][max(i-smooth_num,0):min(i+smooth_num,iter_num)]) for i in range(iter_num)]
    fig,ax1=plt.subplots()
    ax1.plot(iters,smoothed_energy,label='$Energy=%.4f\\pm%.4f$'%(energy_last,sigma_energy_last))
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    if exact_gs_energy!=None:
        plt.axhline(y=exact_gs_energy,c="green",xmin=0,xmax=iters[-1],linewidth=2,label='Exact=%.4f'%(exact_gs_energy))
        delta_energy=numpy.abs(exact_gs_energy-energy_last)
        plt.axis([0,iters[-1],exact_gs_energy-0.1*delta_energy,exact_gs_energy+3*delta_energy])
    ax1.legend()
    if title!=None:
        plt.title(title)
    plt.savefig("./"+name.split(".")[0]+".png")
    plt.show()

if __name__=="__main__":
    help_msg=\
"""a script to show log file of netket, either gui or cui. gui is default.
usage: 
    ./show.py filename --optional_arguments
optional arguments:
    --cui
    --avg_num=1000 the number of batches to be averaged in the output
    --exact=-1.00 exact ground state energy. if provided, this script will draw a line of it in gui mode
    --title=A_Title_for_figure
    --smooth=10 how much the plot will be smoothed
    --help"""
    args={}
    for i in range(1,len(sys.argv)):
        if sys.argv[i].startswith("-"):
            if sys.argv[i].startswith("--cui"):
                args["cui"]=True
            elif sys.argv[i].startswith("--title="):
                args["title"]=sys.argv[i][len("--title="):]
            elif sys.argv[i].startswith("--avg_num="):
                try:
                    args["avg_num"]=int(sys.argv[i][len("--avg_num="):])
                except Exception as e:
                    print("parse argument %s failed: %s"%(sys.argv[i],e))
            elif sys.argv[i].startswith("--smooth="):
                try:
                    args["smooth"]=int(sys.argv[i][len("--smooth="):])
                except Exception as e:
                    print("parse argument %s failed: %s"%(sys.argv[i],e))
            elif sys.argv[i].startswith("--exact="):
                try:
                    args["exact_gs_energy"]=float(sys.argv[i][len("--exact="):])
                except Exception as e:
                    print("parse argument %s failed: %s"%(sys.argv[i],e))
            else:
                print(help_msg)
        else:
            args["filename"]=sys.argv[i]
    
    if "filename" in args:
        if "avg_num" in args:
            avg_num=args["avg_num"]
        else:
            avg_num=None
        if "exact_gs_energy" in args:
            exact_gs_energy=args["exact_gs_energy"]
        else:
            exact_gs_energy=None
        if "title" in args:
            title=args["title"]
        else:
            title=None
        if "smooth" in args:
            smooth=args["smooth"]
        else:
            smooth=None

        if "cui" in args:
            show_data_cui(args["filename"],avg_num=avg_num)
        else:
            show_data(args["filename"],avg_num=avg_num,exact_gs_energy=exact_gs_energy,title=title,smooth_num=smooth)
    else:
        print(help_msg)