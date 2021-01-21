#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from torch_honeycomb import *

RENRM_REAL=2 #DO NOT TOUCH IT! LEAVE IT AS TWO(ER)!

class RBM(nn.Module):
    def __init__(self,state_index,H,parafile=None,device=None):
        super(RBM,self).__init__()
        if device==None:
            self.device=torch.device("cuda:%d"%(random.randint(0,3)))
        else:
            self.device=torch.device(device)

        self.state_index=state_index.to(self.device)
        self.H=H.to(self.device)

        #network topology
        self.hidden_num_1=36
        self.sample_num,self.spinor_num=state_index.shape
        self.w1_re=nn.Linear(self.spinor_num,self.hidden_num_1,bias=False)
        self.w1_im=nn.Linear(self.spinor_num,self.hidden_num_1,bias=False)

        if parafile==None:
            self.rand_init()
        else:
            self.load_para(parafile)
        self.to(self.device)
        log("RBM_3x3 ready: %d hidden nodes, %d parameters, on %s"%(self.hidden_num_1,sum(p.numel() for p in self.parameters()),self.device))

    def rand_init(self,real_bound=0.01,img_bound=math.pi/64):
        real_bound*=RENRM_REAL
        for p in [self.w1_re.weight]:#,self.w1_re.bias]:
            nn.init.uniform_(p,a=-1*real_bound,b=real_bound)
        for p in [self.w1_im.weight]:#,self.w1_im.bias]:
            nn.init.uniform_(p,a=-1*img_bound,b=img_bound)

    def load_para(self,parafile):
        self.load_state_dict(torch.load(parafile,map_location=self.device))

    def imgmul(ax,bx):
        """
            imagin number multiply
                ax, bx: 2-tuple of tensor, denoting real and image part
        """
        ax_re,ax_im=ax
        bx_re,bx_im=bx
        cx_re=ax_re*bx_re - ax_im*bx_im
        cx_im=ax_re*bx_im + ax_im*bx_re
        return (cx_re,cx_im)

    def multiply(self,dx_re,dx_im):
        #the "t" here in tex has no meaning, just for alignment
        """RBM.imgmul((dx_re[:, 41],dx_im[:, 41]),RBM.imgmul((dx_re[:, 40],dx_im[:, 40]),
            RBM.imgmul((dx_re[:, 39],dx_im[:, 39]),RBM.imgmul((dx_re[:, 38],dx_im[:, 38]),
            RBM.imgmul((dx_re[:, 37],dx_im[:, 37]),RBM.imgmul((dx_re[:, 36],dx_im[:, 36]),
            """
        tex=RBM.imgmul((dx_re[:, 35],dx_im[:, 35]),RBM.imgmul((dx_re[:, 34],dx_im[:, 34]),
            RBM.imgmul((dx_re[:, 33],dx_im[:, 33]),RBM.imgmul((dx_re[:, 32],dx_im[:, 32]),
            RBM.imgmul((dx_re[:, 31],dx_im[:, 31]),RBM.imgmul((dx_re[:, 30],dx_im[:, 30]),
            RBM.imgmul((dx_re[:, 29],dx_im[:, 29]),RBM.imgmul((dx_re[:, 28],dx_im[:, 28]),
            RBM.imgmul((dx_re[:, 27],dx_im[:, 27]),RBM.imgmul((dx_re[:, 26],dx_im[:, 26]),
            RBM.imgmul((dx_re[:, 25],dx_im[:, 25]),RBM.imgmul((dx_re[:, 24],dx_im[:, 24]),
            RBM.imgmul((dx_re[:, 23],dx_im[:, 23]),RBM.imgmul((dx_re[:, 22],dx_im[:, 22]),
            RBM.imgmul((dx_re[:, 21],dx_im[:, 21]),RBM.imgmul((dx_re[:, 20],dx_im[:, 20]),
            RBM.imgmul((dx_re[:, 19],dx_im[:, 19]),RBM.imgmul((dx_re[:, 18],dx_im[:, 18]),
            RBM.imgmul((dx_re[:, 17],dx_im[:, 17]),RBM.imgmul((dx_re[:, 16],dx_im[:, 16]),
            RBM.imgmul((dx_re[:, 15],dx_im[:, 15]),RBM.imgmul((dx_re[:, 14],dx_im[:, 14]),
            RBM.imgmul((dx_re[:, 13],dx_im[:, 13]),RBM.imgmul((dx_re[:, 12],dx_im[:, 12]),
            RBM.imgmul((dx_re[:, 11],dx_im[:, 11]),RBM.imgmul((dx_re[:, 10],dx_im[:, 10]),
            RBM.imgmul((dx_re[:, 9],dx_im[:, 9]),RBM.imgmul((dx_re[:, 8],dx_im[:, 8]),
            RBM.imgmul((dx_re[:, 7],dx_im[:, 7]),RBM.imgmul((dx_re[:, 6],dx_im[:, 6]),
            RBM.imgmul((dx_re[:, 5],dx_im[:, 5]),RBM.imgmul((dx_re[:, 4],dx_im[:, 4]),
            RBM.imgmul((dx_re[:, 3],dx_im[:, 3]),RBM.imgmul((dx_re[:, 2],dx_im[:, 2]),
            RBM.imgmul((dx_re[:, 1],dx_im[:, 1]),(dx_re[:,0],dx_im[:,0]))))))))))))))))))))))))))))))))))))
        tex=torch.stack((tex[0],tex[1]))
        return tex

    def pretrain_a(self):
        ws_re=self.w1_re(self.state_index)/RENRM_REAL
        ws_im=self.w1_im(self.state_index)
        dx_re=(1+ws_re.square()/2)*ws_im.cos()
        dx_im=ws_re*ws_im.sin()
        return self.multiply(dx_re,dx_im)

    def stage_a(self):
        ws_re=self.w1_re(self.state_index)/RENRM_REAL
        ws_im=self.w1_im(self.state_index)
        dx_re=ws_re.cosh()*ws_im.cos()
        dx_im=ws_re.sinh()*ws_im.sin()
        return self.multiply(dx_re,dx_im)

    def forward(self):
        if self.train_stage==0:
            bx=self.pretrain_a()
        elif self.train_stage==1:
            bx=self.stage_a()
        else:
            raise Interrupt("stage mistake")
        bx=bx.t()
        bx=self.symmetry.mm(bx)
        norm=bx.square().sum()
        Hv=self.H.mm(bx)
        energy=(bx[:,0].dot(Hv[:,0])+bx[:,1].dot(Hv[:,1]))/norm
        return energy

    def get_overlap(self):
        bx=self.stage_a()
        bx=self.symmetry.mm(bx.t()).t()
        norm=bx.square().sum()
        overlaps=["%.4f"%(100*torch.sqrt(((bx[0,:]*self.gs[i]).sum().square()+(bx[1,:]*self.gs[i]).sum().square())/norm).item()) for i in range(3)]
        return ", ".join(overlaps)

    def train(self):
        #optimizer=optim.SGD(self.parameters(),lr=0.04,momentum=0)
        optimizer=optim.Adam(self.parameters(),lr=0.001,betas=(0.3,0.999),eps=1e-07)
        log("optimizer: %s"%(optimizer.__dict__['defaults'],))
        self.train_stage=0
        for epoch in range(160000+1):
            energy=self()
            if (epoch<5000 and epoch%100==0) or epoch%10000==0:
                prob_list=[]
                with torch.no_grad():
                    #prob_list.append(self.get_overlap())
                    prob_list.append("%.4f"%(self.w1_re.weight.abs().max().item()))
                    prob_list.append("%.4f"%(self.w1_im.weight.abs().max().item()))
                log("%5d: %.8f, %s"%(epoch,energy,prob_list))
            if self.train_stage==0 and epoch==80000:
                self.train_stage=1
                log("increase train stage to 1")
            elif self.train_stage==1 and epoch==120000:
                optimizer=optim.SGD(self.parameters(),lr=0.01,momentum=0)
                log("optimizer: %s"%(optimizer.__dict__['defaults'],))
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
        save_name='Full-%d-%.3f.pkl'%(self.hidden_num_1,self().abs())
        torch.save(self.state_dict(),save_name)
        log("saved state_dict to %s"%(save_name))

import pickle,numpy,itertools

def filp_x(t,nums):
    hidden_num,spinor_num=t.w1_re.weight.shape
    for num in nums:
        t.w1_re.weight[:,num]*=-1
        t.w1_im.weight[:,num]*=-1

def load_matrices(perfix,print=True):
    with open(perfix+".smp",'rb') as f:
        state_index=pickle.load(f)
        log("loaded state_index from %s"%(f.name,))
    with open(perfix+".ha",'rb') as f:
        H=pickle.load(f)
        log("loaded H from %s"%(f.name))
    with open(perfix+".sym",'rb') as f:
        S=pickle.load(f)
        log("loaded S from %s"%(f.name))
    with open(perfix+".eig",'rb') as f:
        eig,eigv=pickle.load(f)
        log("loaded eig from %s: %s ..."%(f.name,eig[0:8]))
    return state_index,H,S,eig,eigv

def main_2():
    with open("3x3_full_right.sym",'rb') as f:
        sym_right=pickle.load(f)
    with open("3x3_full_up.sym",'rb') as f:
        sym_up=pickle.load(f)

    fa=[7,15,1]
    fb=[5,11,13]
    fc=[3,9,17]

    #filp_x(t,(7,))
    bx=t.stage_a().t()
    norm=bx.square().sum()

    Hv=t.H.mm(bx)
    energy=(bx[:,0].dot(Hv[:,0])+bx[:,1].dot(Hv[:,1]))/norm
    log(energy.item())

    bx=bx.t()
    log(["%.4f"%(100*torch.sqrt(((bx[0,:]*t.gs[i]).sum().square()+(bx[1,:]*t.gs[i]).sum().square())/norm).item()) for i in range(3)])

if __name__=="__main__":
    main()