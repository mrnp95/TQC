#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# a script to compute properties of Kitaev's honeycomb model
# author: Sun Youran, Reza

#define consts
#add minus sign here, then you'll never forget minus sign!
JX,JY,JZ=(-1,-1,-1)
#freely tunable parameter, not be used yet
K=0
#three Pauli matrix
sz=[[1,0],[0,-1]]
sx=[[0,1],[1,0]]
sy=[[0,-1j],[1j,0]]
#some flags used in codes
X_FLAG,Y_FLAG,Z_FLAG=(0,1,2)
INV_FLAG=3
#log consts
LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
#log function. I believe my log function is more friendly than logging package
import sys,traceback,math
def log(msg,l=1,end="\n",logfile=None):
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    now_str="%s %03d"%(time.strftime("%y/%m/%d %H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    if l<3:
        tempstr="%s<%s:%d,%s> %s%s"%(now_str,st.name,st.lineno,lstr,str(msg),end)
    else:
        tempstr="%s<%s:%d,%s> %s:\n%s%s"%(now_str,st.name,st.lineno,lstr,str(msg),traceback.format_exc(limit=2),end)
    print(tempstr,end="")
    if l>=1:
        if logfile==None:
            logfile=sys.argv[0].split(".")
            logfile[-1]="log"
            logfile=".".join(logfile)
        with open(logfile,"a") as f:
            f.write(tempstr)

import netket,numpy,time,json

def show_graph(g):
    """show basic infos about a graph"""
    print("graph spinor number: %d, edge number: %d, is_bipartite: %s"%(g.n_sites,len(g.edges),g.is_bipartite))

def show_hilbert(h):
    """show basic infos about a Hilbert space"""
    try:
        print("hilbert size: %d, n_states: %d"%(h.size,h.n_states))
    except Exception as e:
        print(e)
        print("hilbert size: %d"%(h.size))

def get_graph(size,flip=[],show=True):
    """ generate a graph for Kitaev Honeycomb model with double periodic boundry condition
            size: a tuple recording (row_num,col_num)
            flip: records which spin should be fliped, looks like [(0,X_FLAG),(1,Y_FLAG)]
            show: if true, will print some debug infos about the graph
    """
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
              %(len(ex),ex,len(ey),ey,len(ez),ez,),l=0)
    #get graph and show it
    graph=netket.graph.CustomGraph(edges)
    if show:
        show_graph(graph)
    return graph

def get_hamiltonian(size,flip=[],show=True):
    """ generate the hamiltonian for Kitaev Honeycomb model with double periodic boundry condition
            size: a tuple recording (row_num,col_num)
            flip: records which spin should be fliped, looks like [(0,X_FLAG),(1,Y_FLAG)]
            show: if true, will print some debug infos about the graph
    """
    graph=get_graph(size,flip=flip,show=show)
    hilbert=netket.hilbert.Spin(s=0.5,graph=graph)
    hamiltonian=netket.operator.GraphOperator(hilbert
                ,bondops=[(JX*numpy.kron(sx,sx)).tolist(),(JY*numpy.kron(sy,sy)).tolist(),(JZ*numpy.kron(sz,sz)).tolist(),
                          (-1*JX*numpy.kron(sx,sx)).tolist(),(-1*JY*numpy.kron(sy,sy)).tolist(),(-1*JZ*numpy.kron(sz,sz)).tolist()]
                ,bondops_colors=[X_FLAG,Y_FLAG,Z_FLAG,X_FLAG+INV_FLAG,Y_FLAG+INV_FLAG,Z_FLAG+INV_FLAG])
    if show:
        show_hilbert(hilbert)
    return hamiltonian

def exact_diag(hamiltonian,first_n=1,compute_eigenvectors=False):
    """ exact diag hamiltonian
            first_n: how many eigenvalues to compute
            compute_eigenvectors: literal meaning
    """
    tik=time.time()
    res=netket.exact.lanczos_ed(hamiltonian,first_n=first_n,compute_eigenvectors=compute_eigenvectors)
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
    return res

def gs_energy_1(size):
    """ return the groundstate energy
            size: a tuple recording (row_num,col_num)
        it is Babak's version of formula
    """
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

def gs_energy_2(size):
    """ return the groundstate energy
            size: a tuple recording (row_num,col_num)
        it is Youran's version of formula
    """
    Egs=0
    for nx in range(0,size[1]):
        for ny in range(0,size[0]):
            kx=2*numpy.pi*(nx)/size[1]
            ky=2*numpy.pi*(ny)/size[0]
            #kx=2*numpy.pi*(nx+0.5)/size[1]
            #ky=2*numpy.pi*(ny+0.5)/size[0]
            #e=(JX*numpy.cos(kx+ky)+JY*numpy.cos(kx)+JZ*numpy.cos(ky)) #who and when did I wrote this? what does these mean?
            #d=(-1*JX*numpy.sin(kx+ky)+JY*numpy.sin(kx)+JZ*numpy.sin(ky))
            e=(JZ-JX*numpy.cos(kx)-JY*numpy.cos(ky)) #copied from gs_energy_babak
            d=(JX*numpy.sin(kx)+JY*numpy.sin(ky))
            Egs-=numpy.sqrt(e**2+d**2)
    print("Youran's gs energy for %s is %f"%(size,Egs))
    return Egs

def rbm(hamiltonian,machine,output_prefix,n_samples=1000,n_iter=10000,learning_rate=0.01,decay_factor=1):
    """ train rbm using hamiltonian and machine
            output_perfix: filename to save results
    """
    log('machine has %d parameters'%(machine.n_par,))
    log("rbm start with %dkx%dk lr=%.3f df=%.4f"%(n_samples/1000,n_iter/1000,learning_rate,decay_factor))
    sa=netket.sampler.MetropolisLocal(machine=machine)
    op=netket.optimizer.Sgd(learning_rate=0.01,decay_factor=1)
    gs=netket.variational.Vmc(hamiltonian=hamiltonian,sampler=sa,optimizer=op,n_samples=n_samples
                              ,diag_shift=0.1,use_iterative=True,method='Sr')
    start = time.time()
    gs.run(output_prefix=output_prefix,n_iter=n_iter)
    end = time.time()
    log('optimize machine takes %fs'%(end-start,))
    return end-start

def measure(hamiltonian,machine,observables,output_prefix=None,n_iter=100,n_samples=10000):
    """ measure something of a machine
            observables: {"name":observable,...}
    """
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

if __name__=="__main__":
    print("It is honeycomb.py")
    #get_graph((3,3),flip=[(8,0),(8,2)])
    get_graph((6,4))
    #size=(3,3)
    #h=get_hamiltonian(size)
    #print(exact_diag(h))
    #for i in range(2,14):
    #    size=(i,i)
    #    gs_energy_1(size)
    #    gs_energy_2(size)
    #gs_energy_2((3,2))




