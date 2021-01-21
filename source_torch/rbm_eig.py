#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from torch_honeycomb import *
import pickle,sys,numpy,scipy.sparse,scipy.sparse.linalg

def gen_shift_sym():
    with open("3x3_half.smp",'rb') as f:
        state_index=pickle.load(f)
    sample_num,spinor_num=state_index.shape
    S=torch.sparse.FloatTensor(torch.LongTensor([[],[]]),torch.FloatTensor([]),torch.Size([sample_num,sample_num]))
    up_rots,right_rots=([(0,6,14),(1,7,15),(2,8,16),(3,9,17),(4,10,12),(5,11,13)],
                        [(0,2,4),(1,3,5),(6,8,10),(7,9,11),(13,15,17),(12,14,16)])
    S.add_(gen_ops_torchsparse([],[],state_index,rots=right_rots))
    S=S.coalesce()
    log("gened shift symmetry:\n%s"%(S))
    #save
    fname="3x3_half_right.sym"
    with open(fname,'wb') as f:
        pickle.dump(S,f)
        log("successfully dumped S to %s"%(fname))

def torch_sparse_to_scipy(H):
    H=H.coalesce()
    H_indices=H.indices().tolist()
    H_values=H.values().tolist()
    H_size=tuple(H.size())
    H=scipy.sparse.coo_matrix((H_values,(H_indices[0],H_indices[1])),shape=H_size)
    return H

def get_eig():
    perfix="3x3_full_flip"
    with open(perfix+".ha",'rb') as f:
        H=pickle.load(f)
        log("loaded H from %s"%(f.name))
    H=torch_sparse_to_scipy(H)

    log("diaging...")
    eig,eigv=scipy.sparse.linalg.eigs(H,k=16) #cannot find the third -14.29 even k=24 for odd
    log("got eigs: %s"%(eig))
    with open(perfix+".eig",'wb') as f:
        pickle.dump((eig,eigv),f)
        log("dumped to %s"%(f.name))

def eigv_overlap():
    with open("3x3_full_flip2"+".eig","rb") as f:
        eig0,eigv0=pickle.load(f)
        log("load eig0 from %s:\n%s"%(f.name,eig0))
    with open("3x3_full_flip"+".eig","rb") as f:
        eig1,eigv1=pickle.load(f)
        log("load eig1 from %s:\n%s"%(f.name,eig1))

    overlap=numpy.zeros((3,3),dtype=numpy.complex64)
    for i,gi in enumerate([1,3,5]):
        for j,gj in enumerate([0,2,4]):
            assert eig0[gi]<-14.29
            assert eig1[gj]<-14.29
            overlap[i][j]=numpy.vdot(eigv0[:,gi],eigv1[:,gj])
    log("overlap:\n%s"%(overlap))


def translate_eig():
    op_ss=numpy.zeros((3,3),dtype=numpy.complex64)
    for i in range(3):
        #v_bra=numpy.matmul(S,eigv[:,i])
        v_bra=S.dot(eigv[:,i])
        log("v_bra: %s"%(v_bra))
        for j in range(3):
            op_ss[j][i]=numpy.vdot(v_bra,eigv[:,j])
    print(op_ss)
    eig_ss,eigv_ss=numpy.linalg.eig(op_ss)
    print(eig_ss)

if __name__=="__main__":
    #gen_shift_sym()
    #get_eig()
    eigv_overlap()

