#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# a script to compute properties of Kitaev's honeycomb model

import netket,numpy,time,json,sys,traceback

#define consts
JX,JY,JZ=(-1,-1,-1) #add minus sign here, then you'll never forget minus sign!
K=0 #freely tunable parameter
sz=[[1,0],[0,-1]]
sx=[[0,1],[1,0]]
sy=[[0,-1j],[1j,0]]
#sz=sy0
#sx=(-1*numpy.sqrt(3)/2)*numpy.array(sx0)-numpy.array(sy0)*0.5
#sy=(numpy.sqrt(3)/2)*numpy.array(sx0)-numpy.array(sy0)*0.5
X_FLAG,Y_FLAG,Z_FLAG=(0,1,2)
INV_FLAG=3
LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
now_str=lambda: time.strftime("%y/%m/%d %H:%M:%S",time.localtime())

def log(msg,l=0,end="\n",logfile=None):
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    if l<3:
        tempstr="%s<%s:%d,%s> %s%s"%(now_str(),st.name,st.lineno,lstr,str(msg),end)
    else:
        tempstr="%s<%s:%d,%s> %s:\n%s%s"%(now_str(),st.name,st.lineno,lstr,str(msg),traceback.format_exc(limit=2),end)
    print(tempstr,end="")
    if l>=1:
        if logfile==None:
            logfile=sys.argv[0].split(".")
            logfile[-1]="log"
            logfile=".".join(logfile)
        with open(logfile,"a") as f:
            f.write(tempstr)

def show_graph(g):
    print("graph spinor number: %d, edge number: %d, is_bipartite: %s"%(g.n_sites,len(g.edges),g.is_bipartite))
def show_hilbert(h):
    try:
        print("hilbert size: %d, n_states: %d"%(h.size,h.n_states))
    except Exception as e:
        print(e)
        print("hilbert size: %d"%(h.size))
def get_graph_reza(size,flip=[],show=True):
    """ generate a graph for Kitaev Honeycomb model with double periodic boundry condition
        size is a tuple recording (row_num,col_num)
        flip records which spin should be fliped, looks like [(0,X_FLAG),(1,Y_FLAG)]"""
    row_num,col_num=size
    n_lattice=row_num*col_num*2
    edges=[]
    #x&y interactions
    for r in range(row_num):
        begin=r*(2*col_num)
        end=(r+1)*(2*col_num) #exclude
        for o in range(begin+1,end,2):
            if o+1!=end:
                edges.append([o,o+1,X_FLAG])
            else:
                edges.append([o,begin,X_FLAG])
        for e in range(begin,end,2):
            edges.append([e,e+1,Y_FLAG])
    #z interactions
    for o in range(1,n_lattice,2):
        if (n_lattice-2*col_num)<=o:
            pass
        elif int(o/(2*col_num))%2==0:
            #even lines
            edges.append([o,(o+2*col_num-1)%n_lattice,Z_FLAG])
        else:
            #odd lines
            if (o+1)%(4*col_num)!=0:
                #not the one to be sticked
                edges.append([o,(o+2*col_num+1)%n_lattice,Z_FLAG])
            else:
                #the right or left most one to be sticked
                edges.append([o,(o+1)%n_lattice,Z_FLAG])
    #stick upper and lower
    first_line=list(range(0,2*col_num,2))
    last_line=list(range(n_lattice-2*col_num+1,n_lattice,2))
    disloc=int(numpy.ceil(row_num/2)-1)
    for i in range(col_num):
        edges.append([last_line[i],first_line[(i-disloc)%col_num],Z_FLAG])
    #flip spins
    for j in flip:
        for i in edges:
            if (i[0]==j[0] or i[1]==j[0]) and (i[2]%3)==j[1]:
                i[2]=(i[2]+INV_FLAG)%6
                log("fliped %s, %s"%(j,i))
    #show graph
    if show:
        ex=[];ey=[];ez=[]
        for i in edges:
            if i[2]%3==X_FLAG:
                ex.append(i)
            elif i[2]%3==Y_FLAG:
                ey.append(i)
            elif i[2]%3==Z_FLAG:
                ez.append(i)
            else:
                print("error edge: %s"%(i))
        log("\nx_edge number: %d, %s\ny_edge number: %d, %s\nz_edge number: %d, %s"\
              %(len(ex),ex,len(ey),ey,len(ez),ez,))
    #get graph and show it
    graph=netket.graph.CustomGraph(edges)
    if show:
        show_graph(graph)
    return graph
def get_hamiltonian(size,flip=[],show=True):
    """ generate the hamiltonian for Kitaev Honeycomb model with double periodic boundry condition
        size is a tuple recording (row_num,col_num)"""
    graph=get_graph_reza(size,flip=flip,show=show)
    hilbert=netket.hilbert.Spin(s=0.5,graph=graph)
    hamiltonian=netket.operator.GraphOperator(hilbert
                ,bondops=[(JX*numpy.kron(sx,sx)).tolist(),(JY*numpy.kron(sy,sy)).tolist(),(JZ*numpy.kron(sz,sz)).tolist(),
                          (-1*JX*numpy.kron(sx,sx)).tolist(),(-1*JY*numpy.kron(sy,sy)).tolist(),(-1*JZ*numpy.kron(sz,sz)).tolist()]
                ,bondops_colors=[X_FLAG,Y_FLAG,Z_FLAG,X_FLAG+INV_FLAG,Y_FLAG+INV_FLAG,Z_FLAG+INV_FLAG])
    if show:
        show_hilbert(hilbert)
    return hamiltonian
def exact_diag(hamiltonian,first_n=1):
    tik=time.time()
    res=netket.exact.lanczos_ed(hamiltonian,first_n=first_n,compute_eigenvectors=False)
    tok=time.time()
    print("exact_diag used %d s with result:"%(tok-tik))
    TEMP=4
    for i in range(len(res.eigenvalues)):
        value=res.eigenvalues[i]
        print("%14.10f"%(value),end=",")
        if (i+1)%TEMP==0:
            print("")
    else:
        if (i+1)%TEMP!=0:
            print("")
    return res.eigenvalues
def gs_energy_babak(size):
    Egs=0
    for nx in range(-1*(size[1]-1),size[1],2):
        for ny in range(-1*(size[0]-1),size[0],2):
            kx=numpy.pi*nx/size[1]
            ky=numpy.pi*ny/size[0]
            e=(JZ-JX*numpy.cos(kx)-JY*numpy.cos(ky))
            d=(JX*numpy.sin(kx)+JY*numpy.sin(ky))
            Egs-=numpy.sqrt(e**2+d**2)
    print("Babak's gs energy for %s is %f"%(size,Egs))
    return Egs
def gs_energy_mine(size):
    """youran's toy, useless"""
    Egs=0
    for nx in range(0,size[1]):
        for ny in range(0,size[0]):
            kx=2*numpy.pi*(nx)/size[1]
            ky=2*numpy.pi*(ny)/size[0]
            #kx=2*numpy.pi*(nx+0.5)/size[1]
            #ky=2*numpy.pi*(ny+0.5)/size[0]
            e=(JX*numpy.cos(kx+ky)+JY*numpy.cos(kx)+JZ*numpy.cos(ky))
            d=(-1*JX*numpy.sin(kx+ky)+JY*numpy.sin(kx)+JZ*numpy.sin(ky))
            Egs-=numpy.sqrt(e**2+d**2)
    print("My gs energy for %s is %f"%(size,Egs))
    return Egs
def rbm(hamiltonian,machine,output_prefix,n_samples=1000,n_iter=10000):
    sa=netket.sampler.MetropolisLocal(machine=machine)
    op=netket.optimizer.Sgd(learning_rate=0.01,decay_factor=1)
    #op=netket.optimizer.Momentum()
    gs=netket.variational.Vmc(hamiltonian=hamiltonian,sampler=sa,optimizer=op,n_samples=n_samples
                              ,diag_shift=0.1,use_iterative=True,method='Sr')
    print('machine has %d parameters'%(machine.n_par,))
    start = time.time()
    gs.run(output_prefix=output_prefix,n_iter=n_iter)
    end = time.time()
    print('optimize machine takes %fs'%(end-start,))
    return end-start
def measure(hamiltonian,machine,observables,output_prefix=None,delete_temp=True,n_iter=100,n_samples=10000):
    sa=netket.sampler.MetropolisLocal(machine=machine)
    op=netket.optimizer.Sgd(learning_rate=0,decay_factor=0)
    #op=netket.optimizer.Momentum(learning_rate=0,beta=0)
    gs=netket.variational.Vmc(hamiltonian=hamiltonian,sampler=sa,optimizer=op,n_samples=n_samples,diag_shift=0.1)
    for k in observables:
        gs.add_observable(observables[k],str(k))
    if output_prefix==None:
        output_prefix="%f.measuretemp"%(time.time(),)
    elif not output_prefix.endswith(".measuretemp"):
        output_prefix+=".measuretemp"
    log("begin measure %d obervables with %dx%d"%(len(observables),n_samples,n_iter))
    start=time.time()
    gs.run(output_prefix=output_prefix,n_iter=n_iter)
    end=time.time()
    log('measuring %d observables takes %fs'%(len(observables),end-start))
    
    try:
        data=json.load(open(output_prefix+".log"))
    except:
        log("load json failed",l=3)
        return {}
    data_output=data["Output"]
    results={}
    for k in list(observables)+["Energy"]:
        results[k]={}
        results[k]["Mean"]=[i[k]["Mean"] for i in data_output]
        results[k]["Sigma"]=[i[k]["Sigma"] for i in data_output]
        results[k]["Taucorr"]=[i[k]["Taucorr"] for i in data_output]
        try:
            results[k]["mean_Mean"]=numpy.mean(results[k]["Mean"])
            results[k]["mean_Sigma"]=numpy.mean(results[k]["Sigma"])
        except Exception as e:
            print("failed to measure %s: %s"%(k,e))
            results[k]["mean_Mean"]=0
            results[k]["mean_Sigma"]=0
        results[k]["Error"]=results[k]["mean_Sigma"]/numpy.sqrt(len(results[k]["Sigma"]))
    for k in sorted(results):
        log("%s: %f(%f)"%(k,results[k]["mean_Mean"],results[k]["Error"]))
    #if delete_temp:
    #    import os
    #    try:
    #        os.system("rm %s*"%(output_prefix))
    #        print("remove tempdata successfully")
    #    except Exception as e:
    #        print("remove tempdata failed: %s"%(e))
    return results
def flip_yz_spin(para,n_v,n_h,i):
    """just for demo"""
    import copy
    neo_para=copy.deepcopy(para)
    neo_para[i]=-1*neo_para[i]
    for j in range(n_v+n_h+i,len(neo_para),n_v):
        neo_para[j]=-1*neo_para[j]
    return neo_para
def flip_xy_spin(para,n_v,n_h,i):
    """just for demo"""
    import copy
    neo_para=copy.deepcopy(para)
    neo_para[i]=neo_para[i]+numpy.pi*0.5j
    return neo_para
def flip_xz_spin(para,n_v,n_h,i):
    """just for demo"""
    import copy
    neo_para=copy.deepcopy(para)
    neo_para[i]=-1*neo_para[i]+numpy.pi*0.5j
    for j in range(n_v+n_h+i,len(neo_para),n_v):
        neo_para[j]=-1*neo_para[j]
    return neo_para
def flip_spins(machine,n_h,sites):
    """sites looks like ((0,"xy"),(1,"yz"),(2,"zx"))"""
    n_v=machine.n_visible
    log("n_v: %d, n_h: %d, n_par: %d"%(n_v,n_h,machine.n_par))
    assert machine.n_par==n_v+n_h+n_v*n_h,"number of n_h seems incorrect"
    import copy
    neo_para=copy.deepcopy(machine.parameters)
    for i,d in sites:
        if d=="xy" or d=="yx":
            neo_para[i]=neo_para[i]+numpy.pi*0.5j
        elif d=="yz" or d=="zy":
            neo_para[i]*=-1
            for j in range(n_v+n_h+i,len(neo_para),n_v):
                neo_para[j]*=-1
        elif d=="zx" or d=="xz":
            neo_para[i]=-1*neo_para[i]+numpy.pi*0.5j
            for j in range(n_v+n_h+i,len(neo_para),n_v):
                neo_para[j]*=-1
        else:
            print("unrecognized d value: %s"%(d))
    twopi=2*numpy.pi
    for j in range(0,len(neo_para)):
        if abs(neo_para[j].imag)>twopi:
            neo_para[j]=neo_para[j].real+(neo_para[j].imag%twopi)*1j
            print("%s --> %s"%(neo_para[j],temp))
    log("fliped %s"%(sites,))
    return neo_para
def test_flip_spins():
    logfile="gs33.measure"
    hamiltonian=get_hamiltonian((3,3),show=False)
    obs={}
    #for i in range(50):
    for i in range(18):
        obs["sz%d"%(i)]=netket.operator.LocalOperator(hamiltonian.hilbert,[sz,],[[i],])
        obs["sy%d"%(i)]=netket.operator.LocalOperator(hamiltonian.hilbert,[sy,],[[i],])
        obs["sx%d"%(i)]=netket.operator.LocalOperator(hamiltonian.hilbert,[sx,],[[i],])
    machine=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,alpha=2)
    machine.load("./rbm33_a2_1kx10k.wf")
    
    #log("begin measuring unfliped spins",l=1,logfile=logfile)
    #results=measure(hamiltonian,machine,obs,delete_temp=True,n_iter=1000)
    #for k in sorted(results):
    #    log("%s: %f(%f)"%(k,results[k]["mean_Mean"],results[k]["Error"]),l=1,logfile=logfile)
    #return 
    machine.parameters=flip_spins(machine,36,[(0,"xy"),])
    #log("flip 0 in xy and measure again",l=1)
    #results=measure(hamiltonian,machine,obs,delete_temp=True,n_iter=10)
    #for k in sorted(results):
    #    log("%s: %f(%f)"%(k,results[k]["mean_Mean"],results[k]["Error"]),l=1)

    machine.parameters=flip_spins(machine,100,[(19,"yz"),])
    #log("flip 19 in yz and measure again",l=1)
    #results=measure(hamiltonian,machine,obs,delete_temp=True,n_iter=10)
    #for k in sorted(results):
    #    log("%s: %f(%f)"%(k,results[k]["mean_Mean"],results[k]["Error"]),l=1)
     
    machine.parameters=flip_spins(machine,100,[(32,"zx"),])
    #log("flip 32 in zx and measure again",l=1)
    #results=measure(hamiltonian,machine,obs,delete_temp=True,n_iter=10)
    #for k in sorted(results):
    #    log("%s: %f(%f)"%(k,results[k]["mean_Mean"],results[k]["Error"]),l=1)
if __name__=="__main__":
    print("It is honeycomb.py")
    get_graph_reza((3,3),flip=[(8,0),(8,2)])
    #test_flip_spins()
    #print(gs_energy_mine((4,4)))
    

    

