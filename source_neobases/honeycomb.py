#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# a script to compute properties of Kitaev's honeycomb model
# rewrite from /source/honeycomb.py to make it more clear
# and to adapt NetKet 3.0
# author: Sun Youran, Reza

# I believe my log function is more friendly than logging package
LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
import sys,traceback,math,time
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

#define consts
JX,JY,JZ=(-1,-1,-1) #add minus sign here, then you'll never forget minus sign!
K=0                 #freely tunable parameter, not be used yet
sz=[[1,0],[0,-1]]   #three Pauli matrix
sx=[[0,1],[1,0]]
sy=[[0,-1],[1,0]]
X_FLAG,Y_FLAG,Z_FLAG=(0,1,2) #some flags (indices) used in codes
FLIP_SHIFT=3        #X_FLIP_FLAG=X_FLAG+FLIP_SHIFT

import netket,numpy

def show_graph(g):
    """show basic infos about a graph"""
    print("graph spinor number: %d, edge number: %d, is_bipartite: %s"%(g.n_nodes,g.n_edges,g.is_bipartite()))

def show_hilbert(h):
    """show basic infos about a Hilbert space"""
    try:
        print("hilbert size: %d, n_states: %d"%(h.size,h.n_states))
    except Exception as e:
        print(e)
        print("hilbert size: %d"%(h.size))

def get_graph(size,flip=[],show=True):
    """
    generate a graph for Kitaev Honeycomb model with double periodic boundry condition
        size: a tuple recording (row_num,col_num)
        flip: records which spin should be fliped, looks like [(0,X_FLAG),(1,Y_FLAG)]
        show: if true, will print some debug infos about the graph
    """
    row_num,col_num=size
    n_lattice=row_num*col_num*2
    edges=[];rows=[]
    #x&y interactions
    for r in range(row_num):
        begin,end=r*(2*col_num),(r+1)*(2*col_num)
        rows.append(list(range(begin,end)))
        # o for odd and e for even
        edges+=[[o,o+1 if o+1<end else begin,X_FLAG] for o in range(begin+1,end,2)]
        edges+=[[e,e+1,Y_FLAG] for e in range(begin,end,2)]
    #z interactions
    for i,r in enumerate(rows):
        if i<len(rows)-1:
            next_r=rows[i+1]
            bias=-1 if i%2==0 else 1
        else:
            next_r=rows[0]
            bias=-row_num if row_num%2==1 else -row_num+1
        edges+=[[r[j],next_r[(j+bias)%len(r)],Z_FLAG] for j in range(1,len(r),2)]
    #show graph
    if show:
        ex=[i for i in edges if i[2]==X_FLAG]
        ey=[i for i in edges if i[2]==Y_FLAG]
        ez=[i for i in edges if i[2]==Z_FLAG]
        assert len(ex)+len(ey)+len(ez)==len(edges)
        log("\nx_edge number: %d, %s\ny_edge number: %d, %s\nz_edge number: %d, %s"\
              %(len(ex),ex,len(ey),ey,len(ez),ez,),l=0)
    #get graph and show it
    graph=netket.graph.Graph(edges)
    if show:
        show_graph(graph)
    return graph

def get_hamiltonian(size,flip=[],show=True):
    """
    generate the hamiltonian for Kitaev Honeycomb model with double periodic boundry condition
        size: a tuple recording (row_num,col_num)
        flip: records which spin should be fliped, looks like [(0,X_FLAG),(1,Y_FLAG)]
        show: if true, will print some debug infos about the graph
    """
    graph=get_graph(size,flip=flip,show=show)
    hilbert=netket.hilbert.Spin(s=0.5,N=graph.n_nodes)
    bondops=[(JX*numpy.kron(sx,sx)),(JY*numpy.kron(sy,sy)),(JZ*numpy.kron(sz,sz))]
    bondops_colors=[X_FLAG,Y_FLAG,Z_FLAG]
    hamiltonian=netket.operator.GraphOperator(hilbert,graph=graph,bond_ops=bondops,bond_ops_colors=bondops_colors)
    if show:
        show_hilbert(hilbert)
    return hamiltonian

def run_rbm(size,alpha=2,n_samples=4000,n_iter=3000,lr=0.01,df=1):
    """
    train rbm using hamiltonian and machine
        output_perfix: filename to save results
    """
    output_prefix="rbm%d%d_a%d_%dkx%dk"%(size[0],size[1],alpha,n_samples//1000,n_iter//1000)
    log(output_prefix)

    ha=get_hamiltonian(size,show=False)

    """
    sa=netket.sampler.MetropolisLocal(machine=machine)
    op=netket.optimizer.Sgd(learning_rate=lr,decay_factor=df)
    gs=netket.variational.Vmc(hamiltonian=hamiltonian,sampler=sa,optimizer=op,n_samples=n_samples
                              ,diag_shift=0.1,use_iterative=True,method='Sr')"""

    ma=netket.models.RBM(alpha=alpha)
    sa=netket.sampler.MetropolisExchange(hilbert=ha.hilbert,graph=ha.graph)
    vs=netket.vqs.MCState(sa,ma,n_samples=n_samples)
    op=netket.optimizer.Sgd(learning_rate=lr)
    sr=netket.optimizer.SR(diag_shift=0.1)
    gs=netket.VMC(hamiltonian=ha,optimizer=op,preconditioner=sr,variational_state=vs)

    start = time.time()
    gs.run(out=output_prefix,n_iter=n_iter)
    end = time.time()
    log('optimize machine takes %.2fs'%(end-start,))
    return end-start

if __name__=="__main__":
    log("It is /source_neobases/honeycomb.py")
    #get_graph((3,3),flip=[(8,0),(8,2)])
    #get_graph((6,4))
    run_rbm((3,3))