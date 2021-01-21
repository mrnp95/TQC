#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from torch_honeycomb import *
import math

RENRM_REAL=1

class RBM(nn.Module):
    def __init__(self,state_index,H):
        #super(RBM,self).__init__(state_index,H)
        super(RBM,self).__init__()
        #self.a_re=nn.Linear(self.spinor_num,1,bias=False)
        #self.a_im=nn.Linear(self.spinor_num,1,bias=False)

        self.device=torch.device("cpu")
        self.state_index=state_index.to(self.device)
        self.H=H.to(self.device)
        self.sample_num,self.spinor_num=self.state_index.shape
        self.hidden_num_1=5
        self.w1_re=nn.Linear(self.spinor_num,self.hidden_num_1,bias=False)
        self.w1_im=nn.Linear(self.spinor_num,self.hidden_num_1,bias=False)
        """self.hidden_num_2=2
        self.w2_re=nn.Linear(self.spinor_num,self.hidden_num_2,bias=True)
        self.w2_im=nn.Linear(self.spinor_num,self.hidden_num_2,bias=True)
        self.hidden_num_3=2
        self.w3_re=nn.Linear(self.spinor_num,self.hidden_num_3,bias=True)
        self.w3_im=nn.Linear(self.spinor_num,self.hidden_num_3,bias=True)"""

        #for p in [self.w1_re.bias,self.w1_re.weight]:#,self.w2_re.bias,self.w2_re.weight,self.w3_re.bias,self.w3_re.weight]:
        #    nn.init.normal_(p,std=0.02*RENRM_REAL)
        #for p in [self.w1_im.bias,self.w1_im.weight]:#,self.w2_im.bias,self.w2_im.weight,self.w3_im.bias,self.w3_im.weight]:
        #    nn.init.normal_(p,std=math.pi/48)


        #self.para_1=[self.w1_re.bias,self.w1_re.weight,self.w1_im.bias,self.w1_im.weight]
        #self.para_2=[self.w2_re.bias,self.w2_re.weight,self.w2_im.bias,self.w2_im.weight]+self.para_1
        #self.para_3=[self.w3_re.bias,self.w3_re.weight,self.w3_im.bias,self.w3_im.weight]+self.para_2
        self.train_stage=0

        log("I am running on %s"%(self.device))
        self.to(self.device)
        log("I have %d parameters"%(sum(p.numel() for p in self.parameters())))

    def multiply(ax,bx):
        ax_re,ax_im=ax
        bx_re,bx_im=bx
        cx_re=ax_re*bx_re - ax_im*bx_im
        cx_im=ax_re*bx_im + ax_im*bx_re
        return torch.stack((cx_re,cx_im))

    def stage_a(self):
        ws_re=self.w1_re(self.state_index)/RENRM_REAL
        ws_im=self.w1_im(self.state_index)
        dx_re=ws_re.cosh()*ws_im.cos()
        dx_im=ws_re.sinh()*ws_im.sin()
        #print(dx_re)
        #print(dx_im)
        ex=RBM.multiply((dx_re[:,4],dx_im[:,4]),RBM.multiply((dx_re[:,3],dx_im[:,3]),RBM.multiply((dx_re[:,2],dx_im[:,2]),RBM.multiply((dx_re[:,1],dx_im[:,1]),(dx_re[:,0],dx_im[:,0])))))
        #as_re=(self.a_re(self.state_index).view(self.sample_num)/RENRM_REAL).exp()
        #as_re=(self.a_re(self.state_index).view(self.sample_num)/RENRM_REAL).exp()
        #as_im=self.a_im(self.state_index).view(self.sample_num)
        #bx=RBM.multiply(ex,(as_re*as_im.cos(),as_re*as_im.sin()))
        return ex

    def pretrain_a(self):
        ws_re=self.w1_re(self.state_index)/RENRM_REAL
        ws_im=self.w1_im(self.state_index)
        dx_re=(1+ws_re.square())*ws_im.cos()
        dx_im=ws_re*ws_im.sin()
        ex=RBM.multiply((dx_re[:,4],dx_im[:,4]),RBM.multiply((dx_re[:,3],dx_im[:,3]),RBM.multiply((dx_re[:,2],dx_im[:,2]),RBM.multiply((dx_re[:,1],dx_im[:,1]),(dx_re[:,0],dx_im[:,0])))))
        return ex

    """def stage_b(self):
        ws_re=self.w2_re(self.state_index)/RENRM_REAL
        ws_im=self.w2_im(self.state_index)
        dx_re=ws_re.cosh()*ws_im.cos()
        dx_im=ws_re.sinh()*ws_im.sin()
        ex=RBM.multiply((dx_re[:,1],dx_im[:,1]),RBM.multiply((dx_re[:,0],dx_im[:,0]),self.stage_a()))
        return ex

    def stage_c(self):
        ws_re=self.w3_re(self.state_index)/RENRM_REAL
        ws_im=self.w3_im(self.state_index)
        dx_re=ws_re.cosh()*ws_im.cos()
        dx_im=ws_re.sinh()*ws_im.sin()
        ex=RBM.multiply((dx_re[:,1],dx_im[:,1]),RBM.multiply((dx_re[:,0],dx_im[:,0]),self.stage_b()))
        return ex"""

    def forward(self):
        if self.train_stage==0:
            bx=self.pretrain_a()
        elif self.train_stage==1:
            bx=self.stage_a()
        """elif self.train_stage==2:
            bx=self.stage_b()
        elif self.train_stage==3:
            bx=self.stage_c()"""
        return self.calc_energy(bx[0],bx[1])

    def train(self):
        optimizer=optim.SGD(self.para_1,lr=0.02,momentum=0)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[10000,],gamma=0.5)
        for epoch in range(20000):
            energy=self()
            if epoch%100==0:
                #t.w_im.bias t.w_im.weight
                prob_list=[]
                #prob_list.append("%.4f"%(self.w_im.bias.abs().max().item()))
                #prob_list.append(",".join(["%.3f"%(i) for i in self.w_im.bias]))
                prob_list.append("%.8f"%(self.w1_re.weight.abs().max().item()))
                prob_list.append("%.8f"%(self.w1_im.weight.abs().max().item()))
                log("%5d: %.8f, %s"%(epoch,energy,prob_list))
                """if self.train_stage==0 and epoch>=5000:
                    optimizer=optim.SGD(self.para_1,lr=0.04,momentum=0)
                    self.train_stage=1
                    log("add stage to %d"%(self.train_stage))"""
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
            scheduler.step()

import pickle

def main():
    m,n=(2,2)
    state_index=sample(m*n*2)
    #edges=gen_edges((m,n))
    """ex=[i[0:2] for i in edges if i[2]%3==X_FLAG]
    ey=[i[0:2] for i in edges if i[2]%3==Y_FLAG]
    ez=[i[0:2] for i in edges if i[2]%3==Z_FLAG]
    ha_list=[[szsz,ez],[sxsx,ex],[sysy,ey]]
    H=gen_H(state_index,ha_list)
    assert (H==H.transpose(0,1)).sum()==4**(m*n*2)
    with open("2x2.ha",'wb') as f:
        pickle.dump(H,f)
        log("successfully dumped H to 2x2.ha")"""
    with open("2x2.ha",'rb') as f:
        H=pickle.load(f)
        log("successfully loaded H from 2x2.ha")
    t=RBM(state_index,H)
    t.train()

W_RE=torch.tensor(
        [[ 0.33981,  0.35166,  0.30000, -0.28618,  0.28618,  0.30000, -0.35166,  0.33981],
        [      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0],
        [ -0.36870,  0.39058, -0.61698, -0.52412,  0.52412, -0.61698, -0.39058, -0.36870],
        [  0.32355, -0.34563,  0.57544,  0.48047,  0.48047, -0.57544, -0.34563, -0.32355],
        [      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0]])
W_IM=torch.tensor(
        [[   0.76577,    0.78945,   -0.74391,   -0.75921,  -0.026440,   0.041225, -0.0037469,  -0.019920],
        [        0.0,        0.0,        0.0,        0.0, -math.pi/4, -math.pi/4,  math.pi/4,  math.pi/4],
        [    0.27328,   -0.40733,    0.48722,    0.14356,    0.64184,   -0.29819,   -0.37805,   -0.51211],
        [    0.26121,     1.0566,   -0.71337,  -0.021750,   -0.80716,  -0.072030,    0.27116,    0.52419],
        [  math.pi/4,  math.pi/4, -math.pi/4, -math.pi/4, -math.pi/4, -math.pi/4,  math.pi/4,  math.pi/4]])

def refine():
    """refine parameters"""
    state_index=sample(2*2*2)
    with open("2x2.ha",'rb') as f:
        H=pickle.load(f)
        log("successfully loaded H from 2x2.ha")
    t=RBM(state_index,H)
    with torch.no_grad():
        temp=torch.zeros(5,8)
        temp[:,0:4]=W_RE[:,4:8]
        temp[:,4:8]=W_RE[:,0:4]
        t.w1_re.weight.data=temp.to(t.device)
        temp[:,0:4]=W_IM[:,4:8]
        temp[:,4:8]=W_IM[:,0:4]
        t.w1_im.weight.data=temp.to(t.device)

    """t.train_stage=1
    t.para_1=[t.w1_re.weight,t.w1_im.weight]
    t.train()
    print(t.w1_re.weight)
    print(t.w1_im.weight)"""

    #t.state_index=torch.tensor([[-1,-1,1,1,-1,-1,1,1]])
    bx=t.stage_a()
    print(t.calc_energy(bx[0],bx[1]))

    bx=torch.stack([bx[0],bx[1]])
    norm=bx.square().sum()
    bx/=math.sqrt(norm)
    bx=bx.tolist()
    ana_eigv(bx[0])
    ana_eigv(bx[1])

Sx=((0,1),(1,0))
Sy=((0,-1j),(1j,0))
Sz=((1,0),(0,-1))

def overlap():
    state_index=sample_all(2*2*2)
    edges=gen_edges((2,2),show=False)
    H=gen_H(state_index,[(sxsx,[i[0:2] for i in edges if i[2]==X_FLAG]),(sysy,[i[0:2] for i in edges if i[2]==Y_FLAG]),(szsz,[i[0:2] for i in edges if i[2]==Z_FLAG])])
    t=RBM(state_index,H)
    t.w1_re.weight.data=W_RE.to(t.device)
    t.w1_im.weight.data=W_IM.to(t.device)
    with torch.no_grad():
        bx=t.stage_a()

    bx=bx.t()
    norm=bx.square().sum()
    Hv=t.H.mm(bx)
    energy=(bx[:,0].dot(Hv[:,0])+bx[:,1].dot(Hv[:,1]))/norm
    log("energy: %.8f"%(energy))

    import scipy.linalg
    eig,eigv=scipy.linalg.eigh(H,eigvals_only=False) #cannot find the third -14.29 even k=24 for odd
    log("got eigs: %s"%(eig[0:4]))

    gs=eigv[:,0]
    log((gs*gs).sum())
    bx=bx.t()
    log("%.8f"%(100*torch.sqrt(((bx[0,:]*gs).sum().square()+(bx[1,:]*gs).sum().square())/norm).item()))

def fun_a():
    """sum over all spins except one of them"""
    state_index=sample(2*2*2)
    with open("2x2.ha",'rb') as f:
        H=pickle.load(f)
        log("successfully loaded H from 2x2.ha")
    t=RBM(state_index,H)

    with torch.no_grad():
        t.w1_re.weight.data=W_RE.to(t.device)
        t.w1_im.weight.data=W_IM.to(t.device)
    bx=torch.stack(t.stage_a())
    norm=bx.square().sum()
    bx/=math.sqrt(norm)
    #print(bx)
    mask=1<<(7)
    up=0j;down=0j
    for i in range(256):
        amp_temp=abs(bx[0][i].item()+bx[1][i].item()*1j)**2
        if i&mask==0:
            up+=amp_temp
        else:
            down+=amp_temp
    print(up,down)

def test_multiply():
    w_re=torch.rand(5,2)
    w_im=torch.rand(5,2)
    print(w_re)
    print(w_im)
    cx=RBM.multiply((w_re[:,0],w_im[:,0]),(w_re[:,1],w_im[:,1]))
    print(cx)

if __name__=="__main__":
    #fun_a()
    #refine()
    #main()
    overlap()