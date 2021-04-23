#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from torch_honeycomb import *
import torch.nn as nn
import torch.optim as optim
import pickle,numpy,itertools

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

def load_matrices(perfix):
    with open(perfix+".smp",'rb') as f:
        state_index=pickle.load(f)
    # .ha is gened by gen_H_torchsparse in torch_honeycomb
    with open(perfix+".ha",'rb') as f:
        H=pickle.load(f)
    # .sym is gened by gen_shift_sym() in torch_honeycomb
    with open(perfix+".sym",'rb') as f:
        S=pickle.load(f)
    # eig is gened by get_eig in rbm_eig
    with open(perfix+".eig",'rb') as f:
        eig,eigv=pickle.load(f)
    return state_index,H,S,eig,eigv

def main(perfix):
    st,H,S,eig,eigv=load_matrices(perfix)
    #r=RBM(st,H,parafile="Full-36-14.275.pkl",device="cpu")
    r=RBM(st,H,parafile=None,device="cpu")
    r.train_stage=1
    r.symmetry=S.to(r.device)
    r.gs=torch.tensor([eigv[:,1],eigv[:,3],eigv[:,4]],dtype=torch.float32).to(r.device)
    optimizer=optim.SGD(r.parameters(),lr=0.03,momentum=0)
    #optimizer=optim.Adam(r.parameters(),lr=0.001,betas=(0.3,0.999),eps=1e-07)
    log("optimizer: %s"%(optimizer.__dict__['defaults'],))
    ckpts=[-14.275,-14.28,-14.285,-14.29,-14.291,-14.2914,-14.2915]
    ckpt_num=0
    for epoch in range(40*80000+1):
        energy=r()
        if (epoch<1000 and epoch%10==0) or epoch%10000==0:
            prob_list=[]
            with torch.no_grad():
                prob_list.append(r.get_overlap())
                #prob_list.append("%.4f"%(self.w1_re.weight.abs().max().item()))
                #prob_list.append("%.4f"%(self.w1_im.weight.abs().max().item()))
            log("%5d: %.8f, %s"%(epoch,energy,prob_list))
            if energy<ckpts[ckpt_num]: #14.2915
                save_name='Full-%d-%.3f.pkl'%(r.hidden_num_1,r().abs())
                torch.save(r.state_dict(),save_name)
                log("saved state_dict to %s"%(save_name))
                ckpt_num+=1
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()
    save_name='Full-%d-%.3f.pkl'%(r.hidden_num_1,r().abs())
    torch.save(r.state_dict(),save_name)
    log("saved state_dict to %s"%(save_name))

if __name__=="__main__":
    main("operators_3x3/3x3_full")