#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# A frame for honeycomb model using PyTorch
# See rbm2x2.py for example

import time,sys,traceback,math
LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
LOGFILE=sys.argv[0].split(".")
LOGFILE[-1]="log"
LOGFILE=".".join(LOGFILE)
def log(msg,l=1,end="\n",logfile=LOGFILE):
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    now_str="%s.%02d"%(time.strftime("%H:%M:%S",time.localtime()),math.modf(time.time())[0]*100)
    if l<3:
        tempstr="%s [%s,%s:%d] %s%s"%(now_str,lstr,st.name,st.lineno,str(msg),end)
    else:
        tempstr="%s [%s,%s:%d] %s:\n%s%s"%(now_str,lstr,st.name,st.lineno,str(msg),traceback.format_exc(limit=5),end)
    print(tempstr,end="")
    if l>=1:
        with open(logfile,"a") as f:
            f.write(tempstr)

X_FLAG,Y_FLAG,Z_FLAG=(0,1,2)
def gen_edges(size,show=True):
    """
        return the list of [site1,site2,color]
        remember that my numbering rule is different from Reza's
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
    disloc=int(math.ceil(row_num/2)-1)
    for i in range(col_num):
        edges.append([last_line[i],first_line[(i-disloc)%col_num],Z_FLAG])
    #show graph
    if show:
        ex=[i for i in edges if i[2]%3==X_FLAG]
        ey=[i for i in edges if i[2]%3==Y_FLAG]
        ez=[i for i in edges if i[2]%3==Z_FLAG]
        log("\nx_edge number: %d, %s\ny_edge number: %d, %s\nz_edge number: %d, %s"\
              %(len(ex),ex,len(ey),ey,len(ez),ez,),l=0)
    return edges

import torch
import random,numpy,pickle,itertools

def sample_all(spinor_num):
    """
        return list of [1,-1,...] in form of torch array
        sample over the whole Hilbert space
    """
    state_index=torch.ones(2**spinor_num,spinor_num,dtype=torch.float)
    pt_row=1
    for i in range(spinor_num):
        state_index[pt_row:2*pt_row,:]=state_index[0:pt_row,:]
        state_index[pt_row:2*pt_row,i]=-1
        pt_row*=2
    return state_index

def sample_part(spinor_num,sample_num):
    """
        see sample_all
    """
    state_index=torch.randint(0,2,(sample_num,spinor_num),dtype=torch.float64)
    return state_index*2-1

def act_op(op,site,sts):
    """
        op  : like [[0j,1+0j],[1+0j,0j]]
        site: an integer
        sts : like {(1, -1): 1j, (-1, -1): 2j}
    """
    re_sts={}
    for st,wt in sts.items():
        st_fp=list(st)
        st_fp[site]*=-1
        st_fp=tuple(st_fp)
        if st[site]==1:
            if op[0][0]!=0:
                if st in re_sts:
                    re_sts[st]+=wt*op[0][0]
                else:
                    re_sts[st]=wt*op[0][0]
            if op[1][0]!=0:
                if st_fp in re_sts:
                    re_sts[st_fp]+=wt*op[1][0]
                else:
                    re_sts[st_fp]=wt*op[1][0]
        elif st[site]==-1:
            if op[1][1]!=0:
                if st in re_sts:
                    re_sts[st]+=wt*op[1][1]
                else:
                    re_sts[st]=wt*op[1][1]
            if op[0][1]!=0:
                if st_fp in re_sts:
                    re_sts[st_fp]+=wt*op[0][1]
                else:
                    re_sts[st_fp]=wt*op[0][1]
        else:
            raise Exception("in act_op")
    return re_sts

def act_rot(st,rots):
    """
        act rotation
            st    : like (1, -1)
            rots: list of rot
                rot: like [1,2,3](abc --> bca)
    """
    re_st=list(st)
    for rot in rots:
        temp=re_st[rot[0]]
        for i in range(len(rot)-1):
            re_st[rot[i]]=re_st[rot[i+1]]
        re_st[rot[-1]]=temp
    return tuple(re_st)

def gen_ops_npmat(ops,sites,state_index,rots=None):
    """
        generate operators in numpy matrix form
            ops        : list of op which will be passed to act_op; can be empty
            sites      : list of site which will be passed to act_op
            state_index: like[[ 1., 1., ..., 1., 1.], ... ]
    """
    assert len(ops)==len(sites)
    sample_num,spinor_num=state_index.shape
    op_mat=numpy.zeros((sample_num,sample_num),dtype=numpy.complex64)
    for i,st in enumerate(state_index):
        sts_out={tuple(st.tolist()):1}
        for alpha,op in enumerate(ops):
            sts_out=act_op(op,sites[alpha],sts_out)
        for st_out,wt in sts_out.items():
            if rots!=None:
                st_out=act_rot(st_out,rots)
            st_out=torch.tensor(st_out)
            v,j=(st_out==state_index).sum(1).max(0)
            #assert v==spinor_num, "neost is not in state_index"
            if v==spinor_num:
                op_mat[j][i]=wt
    return op_mat

def gen_ops_torchsparse(ops,sites,state_index,rots=None):
    """
        like gen_ops_npmat but return pytorch sparse matrix
    """
    assert len(ops)==len(sites)
    sample_num,spinor_num=state_index.shape
    indices=[[0,0]] #saves the indices of non-zeros entries
    values=[0] #values
    for i,st in enumerate(state_index):
        sts_out={tuple(st.tolist()):1}
        for alpha in range(len(ops)):
            sts_out=act_op(ops[alpha],sites[alpha],sts_out)
        for st_out,wt in sts_out.items():
            if rots!=None:
                st_out=act_rot(st_out,rots)
            st_out=torch.tensor(st_out)
            v,j=(st_out==state_index).sum(1).max(0)
            assert v==spinor_num, "neost is not in state_index"
            indices.append([j,i])
            assert wt.imag==0, "wt is not real: %s"%(wt)
            values.append(wt.real)
    indices=torch.LongTensor(indices).t()
    values=torch.FloatTensor(values)
    opmat=torch.sparse.FloatTensor(indices,values,torch.Size([sample_num,sample_num]))
    return opmat

def szsz(state,edge):
    """
        state is a N-list
        edge is a M-list, where M is the number of interactions
    """
    neost=state.clone()
    if (neost[edge[0]]==1 and neost[edge[1]]==1) or (neost[edge[0]]==-1 and neost[edge[1]]==-1):
        return neost,1
    else:
        return neost,-1

def sxsx(state,edge):
    """see szsz"""
    neost=state.clone()
    neost[edge[0]]*=-1
    neost[edge[1]]*=-1
    return neost,1

def sysy(state,edge):
    """see szsz"""
    neost=state.clone()
    neost[edge[0]]*=-1
    neost[edge[1]]*=-1
    if (neost[edge[0]]==1 and neost[edge[1]]==1) or (neost[edge[0]]==-1 and neost[edge[1]]==-1):
        return neost,-1
    else:
        return neost,1

def gen_H(state_index,ha_list):
    """
        ha_list is a list of (function,edges)
        where 'function' takes edge and a state as input and gives out_state,weight
        state_index is higher end, i.e. 0b10...0 is the first spiner
    """
    sample_num,spinor_num=state_index.shape
    H=torch.zeros(sample_num,sample_num,dtype=torch.float32)
    for i,sti in enumerate(state_index):
        for h_part,edges in ha_list:
            for e in edges:
                neost,weight=h_part(sti,e)
                v,j=(neost==state_index).sum(1).max(0)
                #assert v==spinor_num
                if v==spinor_num:
                    H[j,i]+=weight
    #remember times minus one!!!
    H*=-1
    return H

Sx=((0,1),(1,0))
Sy=((0,-1j),(1j,0))
Sz=((1,0),(0,-1))

def gen_H_torchsparse(perfix):
    #load samples
    smp_file=perfix+".smp"
    with open(smp_file,'rb') as f:
        state_index=pickle.load(f)
        log("successfully loaded state_index from %s"%(smp_file))
    #generate Hamiltonian
    sample_num,spinor_num=state_index.shape
    H=torch.sparse.FloatTensor(torch.LongTensor([[],[]]),torch.FloatTensor([]),torch.Size([sample_num,sample_num]))
    for i in gen_edges((3,3),show=True):
        log("dealing edge for %s"%(i,))
        if i[2]%3==X_FLAG:
            mat_temp=gen_ops_torchsparse([Sx,Sx],i[0:2],state_index)
        elif i[2]%3==Y_FLAG:
            mat_temp=gen_ops_torchsparse([Sy,Sy],i[0:2],state_index)
        elif i[2]%3==Z_FLAG:
            mat_temp=gen_ops_torchsparse([Sz,Sz],i[0:2],state_index)
        H.add_(mat_temp)
    if not H.is_coalesced():
        H=H.coalesce()
    #remember to multiply -1!
    H*=-1
    assert (H-H.transpose(0,1)).coalesce()._values().abs().sum()==0, "H is not Hermitian!"
    #save
    fname=perfix+".ha"
    with open(fname,'wb') as f:
        pickle.dump(H,f)
        log("successfully dumped H to %s"%(fname))

def gen_shift_sym(perfix):
    #load samples
    smp_file=perfix+".smp"
    with open(smp_file,'rb') as f:
        state_index=pickle.load(f)
        log("successfully loaded state_index from %s"%(smp_file))
    sample_num,spinor_num=state_index.shape
    S=torch.sparse.FloatTensor(torch.LongTensor([[],[]]),torch.FloatTensor([]),torch.Size([sample_num,sample_num]))
    up_rots=[(0,6,14),(1,7,15),(2,8,16),(3,9,17),(4,10,12),(5,11,13)]
    right_rots=[(0,2,4),(1,3,5),(6,8,10),(7,9,11),(13,15,17),(12,14,16)]
    for i,j in itertools.product(range(3),range(3)):
        log("generating (%d,%d)th symmetry"%(i,j))
        S.add_(gen_ops_torchsparse([],[],state_index,rots=up_rots*i+right_rots*j))
    S=S.coalesce()
    log("gened shift symmetry:\n%s"%(S))
    #save
    fname=perfix+".sym"
    with open(fname,'wb') as f:
        pickle.dump(S,f)
        log("successfully dumped S to %s"%(fname))

if __name__=="__main__":
    pass