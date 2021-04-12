#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from torch_honeycomb import *
import math,pickle

RENRM_REAL=2

class RBM(nn.Module):
    def __init__(self,state_index,H):
        super(RBM,self).__init__()

        self.device=torch.device("cpu")
        self.H=H.to(self.device)
        self.state_index=state_index.to(self.device)
        self.sample_num,self.spinor_num=self.state_index.shape

        self.hidden_num_1=5
        self.w1_re=nn.Linear(self.spinor_num,self.hidden_num_1,bias=False)
        self.w1_im=nn.Linear(self.spinor_num,self.hidden_num_1,bias=False)
        #If you need bias
        #self.a_re=nn.Linear(self.spinor_num,1,bias=False)
        #self.a_im=nn.Linear(self.spinor_num,1,bias=False)

        #If you want to init manully
        #for p in [self.w1_re.bias,self.w1_re.weight]:#,self.w2_re.bias,self.w2_re.weight,self.w3_re.bias,self.w3_re.weight]:
        #    nn.init.normal_(p,std=0.02*RENRM_REAL)
        #for p in [self.w1_im.bias,self.w1_im.weight]:#,self.w2_im.bias,self.w2_im.weight,self.w3_im.bias,self.w3_im.weight]:
        #    nn.init.normal_(p,std=math.pi/48)

        self.para_1=[self.w1_re.weight,self.w1_im.weight] #,self.w1_re.bias,self.w1_im.bias]
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
        ex= RBM.multiply((dx_re[:,4],dx_im[:,4]),RBM.multiply((dx_re[:,3],dx_im[:,3]),
            RBM.multiply((dx_re[:,2],dx_im[:,2]),RBM.multiply((dx_re[:,1],dx_im[:,1]),(dx_re[:,0],dx_im[:,0])))))
        return ex

    def pretrain_a(self):
        ws_re=self.w1_re(self.state_index)/RENRM_REAL
        ws_im=self.w1_im(self.state_index)
        dx_re=(1+ws_re.square())*ws_im.cos()
        dx_im=ws_re*ws_im.sin()
        ex=RBM.multiply((dx_re[:,4],dx_im[:,4]),RBM.multiply((dx_re[:,3],dx_im[:,3]),RBM.multiply((dx_re[:,2],dx_im[:,2]),RBM.multiply((dx_re[:,1],dx_im[:,1]),(dx_re[:,0],dx_im[:,0])))))
        return ex

    def forward(self):
        if self.train_stage==0:
            bx=self.pretrain_a()
        elif self.train_stage==1:
            bx=self.stage_a()
        bx=bx.t()
        norm=bx.square().sum()
        Hv=self.H.mm(bx)
        energy=(bx[:,0].dot(Hv[:,0])+bx[:,1].dot(Hv[:,1]))/norm
        return energy

    def train(self):
        optimizer=optim.SGD(self.para_1,lr=0.02,momentum=0)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[10000,],gamma=0.5)
        for epoch in range(20000):
            energy=self()
            if epoch%100==0:
                prob_list=[]
                #prob_list.append("%.4f"%(self.w_im.bias.abs().max().item()))
                #prob_list.append(",".join(["%.3f"%(i) for i in self.w_im.bias]))
                prob_list.append("%.8f"%(self.w1_re.weight.abs().max().item()))
                prob_list.append("%.8f"%(self.w1_im.weight.abs().max().item()))
                log("%5d: %.8f, %s"%(epoch,energy,prob_list))
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
            scheduler.step()

def main():
    """
        optimize a 2x2 Honeycomb model
    """
    m,n=(2,2)
    state_index=sample_all(m*n*2)

    generate_H_flag=False
    if generate_H_flag:
        edges=gen_edges((m,n))
        ex=[i[0:2] for i in edges if i[2]%3==X_FLAG]
        ey=[i[0:2] for i in edges if i[2]%3==Y_FLAG]
        ez=[i[0:2] for i in edges if i[2]%3==Z_FLAG]
        ha_list=[[szsz,ez],[sxsx,ex],[sysy,ey]]
        H=gen_H(state_index,ha_list)
        assert (H==H.transpose(0,1)).sum()==4**(m*n*2)
        with open("2x2.ha",'wb') as f:
            pickle.dump(H,f)
            log("successfully dumped H to 2x2.ha")
    else:
        with open("2x2.ha",'rb') as f:
            H=pickle.load(f)
            log("successfully loaded H from 2x2.ha")

    t=RBM(state_index,H)
    t.train()

"""
An ideal result look like
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
"""

def test_multiply():
    w_re=torch.rand(5,2)
    w_im=torch.rand(5,2)
    print(w_re)
    print(w_im)
    cx=RBM.multiply((w_re[:,0],w_im[:,0]),(w_re[:,1],w_im[:,1]))
    print(cx)

if __name__=="__main__":
    #test_multiply()
    main()