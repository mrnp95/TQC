#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# a script to compute properties of Kitaev's honeycomb model

import netket,numpy,time,json,sys
import matplotlib.pyplot as plt

#define consts
L_X=4
L_Y=4
JX,JY,JZ=(-1,-1,-1) #add minus sign here, then you'll never forget minus sign!
K=0 #freely tunable parameter
sz=[[1,0],[0,-1]]
sx=[[0,1],[1,0]]
sy=[[0,-1j],[1j,0]]
#sz=sy0
#sx=(-1*numpy.sqrt(3)/2)*numpy.array(sx0)-numpy.array(sy0)*0.5
#sy=(numpy.sqrt(3)/2)*numpy.array(sx0)-numpy.array(sy0)*0.5
X_FLAG,Y_FLAG,Z_FLAG=(0,1,2)

def show_graph(g):
    print("graph spinor number: %d, edge number: %d, is_bipartite: %s"%(g.n_sites,len(g.edges),g.is_bipartite))
def show_hilbert(h):
    try:
        print("hilbert size: %d, n_states: %d"%(h.size,h.n_states))
    except Exception as e:
        print(e)
        print("hilbert size: %d"%(h.size))
def get_graph_obsolete(size,show=True):
    """size is an tuple recording (row_num,col_num), row_num should be an even number"""
    assert size[0]%2==0,"size[0], i.e.row_num should be even"
    row_num,col_num=size
    n_lattice=row_num*col_num*2
    edges=[]
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
    for o in range(1,n_lattice,2):
        if int(o/(2*col_num))%2==0:
            edges.append([o,(o+2*col_num-1)%n_lattice,Z_FLAG])
        else:
            if (o+1)%(4*col_num)!=0:
                edges.append([o,(o+2*col_num+1)%n_lattice,Z_FLAG])
            else:
                edges.append([o,(o+1)%n_lattice,Z_FLAG])
    
    graph=netket.graph.CustomGraph(edges)
    if show:
        show_graph(graph)
        ex=[];ey=[];ez=[]
        for i in edges:
            if i[2]==X_FLAG:
                ex.append(i[0:2])
            elif i[2]==Y_FLAG:
                ey.append(i[0:2])
            elif i[2]==Z_FLAG:
                ez.append(i[0:2])
            else:
                print("error edge: %s"%(i))
        print("x_edge number: %d, %s\ny_edge number: %d, %s\nz_edge number: %d, %s"\
              %(len(ex),ex,len(ey),ey,len(ez),ez,))
    return graph
def get_graph_reza(size,show=True):
    """ generate a graph for Kitaev Honeycomb model with double periodic boundry condition
        size is a tuple recording (row_num,col_num)"""
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
    if show:
        ex=[];ey=[];ez=[]
        for i in edges:
            if i[2]==X_FLAG:
                ex.append(i[0:2])
            elif i[2]==Y_FLAG:
                ey.append(i[0:2])
            elif i[2]==Z_FLAG:
                ez.append(i[0:2])
            else:
                print("error edge: %s"%(i))
        print("x_edge number: %d, %s\ny_edge number: %d, %s\nz_edge number: %d, %s"\
              %(len(ex),ex,len(ey),ey,len(ez),ez,))
    #get graph and show it
    graph=netket.graph.CustomGraph(edges)
    if show:
        show_graph(graph)
    return graph

def get_hamiltonian(size,show=True):
    """ generate the hamiltonian for Kitaev Honeycomb model with double periodic boundry condition
        size is a tuple recording (row_num,col_num)"""
    graph=get_graph_reza(size,show=show)
    hilbert=netket.hilbert.Spin(s=0.5,graph=graph)
    hamiltonian=netket.operator.GraphOperator(hilbert
                ,bondops=[(JX*numpy.kron(sx,sx)).tolist(),(JY*numpy.kron(sy,sy)).tolist(),(JZ*numpy.kron(sz,sz)).tolist()]
                ,bondops_colors=[X_FLAG,Y_FLAG,Z_FLAG])
    show_hilbert(hilbert)
    return hamiltonian
def get_hamiltonian_majorana(size,show=True):
    """unuseable"""
    c=[[0,1],[1,0]]
    a=1j
    graph=get_graph_reza(size,show=show)
    hilbert=netket.hilbert.Boson(graph=graph,n_max=1)
    hamiltonian=netket.operator.GraphOperator(hilbert
                ,bondops=[(-1*a*JX*numpy.kron(c,c)).tolist(),(a*JY*numpy.kron(c,c)).tolist(),(-1*a*JZ*numpy.kron(c,c)).tolist()]
                ,bondops_colors=[X_FLAG,Y_FLAG,Z_FLAG])
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
    Egs=0
    for nx in range(0,size[1]):
        for ny in range(0,size[0]):
            kx=2*numpy.pi*(nx+0.5)/size[1]
            ky=2*numpy.pi*(ny+0.5)/size[0]
            e=(JX*numpy.cos(kx+ky)+JY*numpy.cos(kx)+JZ*numpy.cos(ky))
            d=(-1*JX*numpy.sin(kx+ky)+JY*numpy.sin(kx)+JZ*numpy.sin(ky))
            Egs-=numpy.sqrt(e**2+d**2)
    print("My gs energy for %s is %f"%(size,Egs))
    return Egs
def rbm(hamiltonian):
    #ma=netket.machine.RbmSpin(alpha=1,hilbert=hamiltonian.hilbert,use_visible_bias=True)
    ma=netket.machine.RbmSpin(n_hidden=36,hilbert=hamiltonian.hilbert,use_visible_bias=True)
    ma.init_random_parameters(sigma=0.01)

    sa=netket.sampler.MetropolisLocal(machine=ma)
    
    op=netket.optimizer.Sgd(learning_rate=0.01,decay_factor=1)
    #op=netket.optimizer.Momentum()
    gs=netket.variational.Vmc(hamiltonian=hamiltonian,sampler=sa,optimizer=op,n_samples=400,diag_shift=0.1,use_iterative=True,method='Sr')
    #print(help(op))
    #return
    print('RBM calculation has %d parameters'%(ma.n_par,))
    start = time.time()
    gs.run(output_prefix='rbm66_1101_7', n_iter=10000)
    end = time.time()
    print('RBM calculation takes %fs'%(end-start,))
def find_min(energys):
    neo=sorted(energys)
    print("minimum energy is %s"%(neo[0:2]))
def show_data_cui(name="RBM.log"):
    skipped_obs=("Iteration","Acceptance","EnergyVariance")
    data=json.load(open(name))
    iters=[]
    energys=[]
    
    for i in data["Output"]:
        iters.append(i["Iteration"])
        energys.append(i["Energy"]["Mean"])
    TEMP=100
    for i in range(0,len(iters),TEMP):
        print("%3d-%3d: %14.10f %14.10f"%(i,i+TEMP-1,numpy.mean(energys[i:i+TEMP-1]),numpy.std(energys[i:i+TEMP-1])))
    find_min(energys)
def show_data(name,exact_gs_energy=None):
    data=json.load(open(name))
    # Extract the relevant information
    skipped_obs=("Iteration","Acceptance","EnergyVariance")
    iters=[]
    iters=[]
    energys=[]
    for i in data["Output"]:
        iters.append(i["Iteration"])
        energys.append(i["Energy"]["Mean"])
    fig,ax1=plt.subplots()
    ax1.plot(iters,energys,label='Energy')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    if exact_gs_energy!=None:
        plt.axhline(y=exact_gs_energy,c="green",xmin=0,xmax=iters[-1],linewidth=2,label='Exact')
        plt.axis([0,iters[-1],exact_gs_energy-1,exact_gs_energy+10])
    ax1.legend(loc=2)
    plt.title("Sgd,400*10k,learn=0.01,360nodes")
    plt.savefig("./"+name.split(".")[0]+".png")
    plt.show()
    find_min(energys)
    
if __name__=="__main__":
    #get_graph_reza((4,3))
    #gs_energy_babak((4,3))
    size=(2,2)
    
    if len(sys.argv)>=2 and sys.argv[1]=="show":
        if len(sys.argv)==3:
            show_data(sys.argv[2])
        elif len(sys.argv)==4:
            if sys.argv[3]=="cui":
                show_data_cui(sys.argv[2])
            else:
                show_data(sys.argv[2],eval(sys.argv[3]))
        else:
            show_data_cui()
    elif len(sys.argv)>=2 and sys.argv[1]=="diag":
        h=get_hamiltonian(size)
        exact_diag(h,first_n=8)
    elif len(sys.argv)>=3 and sys.argv[1]=="exact":
        gs_energy_babak(eval(sys.argv[2]))
        gs_energy_mine(eval(sys.argv[2]))
    else:
        h=get_hamiltonian(size)
        rbm(h)
    

    

