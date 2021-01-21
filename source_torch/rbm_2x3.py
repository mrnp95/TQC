#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from torch_honeycomb import *
import math

RENRM_REAL=2

class RBM(TorchHoneycomb):
    def __init__(self,state_index,H,device=None):
        super(RBM,self).__init__(state_index,H,device)
        self.hidden_num_1=6
        self.w1_re=nn.Linear(self.spinor_num,self.hidden_num_1,bias=False)
        self.w1_im=nn.Linear(self.spinor_num,self.hidden_num_1,bias=False)
        """self.hidden_num_2=4
        self.w2_re=nn.Linear(self.spinor_num,self.hidden_num_2,bias=False)
        self.w2_im=nn.Linear(self.spinor_num,self.hidden_num_2,bias=False)"""

        for p in [self.w1_re.weight,]:
            #nn.init.normal_(p,std=0.02*RENRM_REAL)
            ubnd=0.03*RENRM_REAL
            nn.init.uniform_(p,a=-1*ubnd,b=ubnd)
        for p in [self.w1_im.weight,]:
            #nn.init.normal_(p,std=math.pi/48)
            ubnd=math.pi/32
            nn.init.uniform_(p,a=-1*ubnd,b=ubnd)
        """for p in [self.w2_re.weight,self.w2_im.weight]:
            nn.init.zeros_(p)"""

        self.para_1=[self.w1_re.weight,self.w1_im.weight]
        #self.para_2=[self.w2_re.weight,self.w2_im.weight]
        self.train_stage=0

        log("I am running on %s"%(self.device))
        self.to(self.device)
        log("I have %d parameters"%(sum(p.numel() for p in self.parameters())))

        """self.plq_mask=torch.tensor([[1,1,0,0,0,1,1,0,0,0,1,1],[0,1,1,1,0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0,1,1,1,0],
                                    [1,0,0,0,1,1,1,1,0,0,0,1],[1,1,1,0,0,0,0,1,1,1,0,0],[0,0,1,1,1,0,0,0,0,1,1,1]],device=self.device)
        self.dia_mask=torch.tensor([[1,0,1,0,0,0,0,1,0,0,0,1],[1,0,0,0,1,0,0,0,0,1,0,1],[0,0,1,0,1,0,0,1,0,1,0,0],
                                    [0,1,0,1,0,0,0,0,1,0,1,0],[0,1,0,0,0,1,1,0,1,0,0,0],[0,0,0,1,0,1,1,0,0,0,1,0]],device=self.device)
        self.restrict_para()"""

    """def restrict_para(self):
        with torch.no_grad():
            self.w1_im.weight.data[0:6,:]*=self.plq_mask"""

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
        """
        #multiply by first taking log then exp
        ex_rho=((dx_re.square()+dx_im.square()).log()).sum(1)
        ex_rho=((ex_rho+self.symmetry.matmul(ex_rho))/2).exp()
        ex_phi=(dx_im/(dx_re)).atan().sum(1)
        ex_phi=(ex_phi+self.symmetry.matmul(ex_phi))
        return (ex_rho*ex_phi.cos(),ex_rho*ex_phi.sin())
        """
        #the "t" here in tex has no meaning, just for alignment
        tex=RBM.imgmul((dx_re[:, 5],dx_im[:, 5]),RBM.imgmul((dx_re[:, 4],dx_im[:, 4]),
            RBM.imgmul((dx_re[:, 3],dx_im[:, 3]),RBM.imgmul((dx_re[:, 2],dx_im[:, 2]),
            RBM.imgmul((dx_re[:, 1],dx_im[:, 1]),(dx_re[:,0],dx_im[:,0]))))))
        #symmetry by multiply
        #tex=RBM.imgmul(tex,(self.symmetry.matmul(tex[0]),self.symmetry.matmul(tex[1])))
        #symmetry by sum
        tex=torch.stack((tex[0],tex[1]))
        tex=tex.matmul(self.symmetry)
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

    """def pretrain_b(self):
        #TODO!
        ws_re=torch.cat((self.w1_re(self.state_index),self.w2_re(self.state_index)),1)/RENRM_REAL
        ws_im=torch.cat((self.w1_im(self.state_index),self.w2_im(self.state_index)),1)
        dx_re=(1+ws_re.square()/2)*ws_im.cos()
        dx_im=ws_re*ws_im.sin()
        return RBM.multiply(dx_re,dx_im)"""

    """def stage_b(self):
        ws_re=torch.cat((self.w1_re(self.state_index),self.w2_re(self.state_index)),1)/RENRM_REAL
        ws_im=torch.cat((self.w1_im(self.state_index),self.w2_im(self.state_index)),1)
        dx_re=ws_re.cosh()*ws_im.cos()
        dx_im=ws_re.sinh()*ws_im.sin()
        return RBM.multiply(dx_re,dx_im)"""

    def forward(self):
        if self.train_stage==0:
            bx=self.pretrain_a()
        elif self.train_stage==1:
            bx=self.stage_a()
        #elif self.train_stage==2:
        #    bx=self.stage_b()
        else:
            raise Interrupt
        return self.calc_energy(bx)

    def dst(v,w):
        """calc dst v w.r.t. w"""
        return ((w*50).round().sign()-v.sign()).square().sum().item()

    def train(self):
        optimizer=optim.SGD(self.para_1,lr=0.12,momentum=0)
        #scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[1,],gamma=0.5)
        last_st=0
        for epoch in range(30000+1):
            energy=self()
            if epoch%1000==0:
                prob_list=[]
                with torch.no_grad():
                    ws_re=self.w1_re(self.state_index)/RENRM_REAL
                    ws_im=self.w1_im(self.state_index)
                    dx_re=ws_re.cosh()*ws_im.cos()
                    dx_im=ws_re.sinh()*ws_im.sin()
                    prob_list.append("%.4f"%(dx_re.abs().max().item()))
                    prob_list.append("%.2e"%(dx_re.abs().min().item()))
                log("%5d: %.8f, %s"%(epoch,energy,prob_list))
                if self.train_stage==0 and epoch==10000:
                    optimizer=optim.SGD(self.para_1,lr=0.06,momentum=0)
                elif self.train_stage==0 and epoch==20000:
                    st="stage_a"
                    optimizer=optim.SGD(self.para_1,lr=0.04,momentum=0)
                    self.train_stage+=1;mst=epoch
                #elif self.train_stage==1 and epoch==30000:
                #    optimizer=optim.SGD(self.para_1,lr=0.02,momentum=0)
                if last_st!=self.train_stage:
                    log("add stage to %d: %s"%(self.train_stage,st))
                    last_st=self.train_stage
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
            #scheduler.step()
            #self.restrict_para()
        print(self.w1_re.weight.data)
        print(self.w1_im.weight.data)
        print("%.8f"%(self()))

import pickle,sys,numpy

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
            raise Exception
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
            assert v==spinor_num, "neost is not in state_index"
            op_mat[j][i]=wt
    return op_mat

def test_gen_ops_npmat():
    I=numpy.array([[1,0j],[0j,1]])
    Sx=numpy.array([[0j,1+0j],[1+0j,0j]])
    Sy=numpy.array([[0j,-1j],[1j,0j]])
    Sz=numpy.array([[1+0j,0j],[0j,-1+0j]])
    C6=(I+1j*Sx+1j*Sy+1j*Sz)/2 #[[(1+1j)/2,(1+1j)/2],[(-1+1j)/2,(1-1j)/2]]

    #test act_op
    """sts={(1,-1,0.5):1j};print(sts)
    sts=act_op([[1,0],[2,0]],0,sts);print(sts)
    sts=act_op([[3,5],[7,11]],0,sts);print(sts)
    sts=act_op([[5,6],[7,8]],1,sts);print(sts)"""
    #test act_rot
    #print(act_rot((1,2,3,4,5),[(1,2,4),]))

    fname="2x3_part.ha"
    with open(fname,'rb') as f:
        state_index,H=pickle.load(f)
        log("successfully loaded (state_index,H) from %s"%(fname))

    #mat=gen_ops_npmat([Sy,Sz,Sx,Sy,Sz,Sx],[3,4,5,10,9,8],state_index) # a plaqette operator for 2x3
    #mat=gen_ops_npmat([C6]*8,list(range(8)),state_index,rots=[(1,5,3),(2,4,6)]) # C6 for 2x2
    #mat=gen_ops_npmat([],[],state_index,rots=[(0,4,2),(1,5,3),(6,10,8),(7,11,9)]) # plaqette left translation for 2x3
    mat=gen_ops_npmat([],[],state_index,rots=[(0,6),(1,7),(2,8),(3,9),(4,10),(5,11)]) # left-right translation for 2x3
    #numpy.identity(2048)
    #print(numpy.array_equal(numpy.matmul(numpy.matmul(matb,mata),matb),mata))

    first_n=4 #ground state for 2x3 is 4-fold degenerate!
    eig,eigv=numpy.linalg.eigh(H.numpy())
    print("%.8f"%(eig[0]))
    print("%.8f"%(eig[1]))
    op_ss=numpy.zeros((first_n,first_n),dtype=numpy.complex64)
    for i in range(first_n):
        v_bra=numpy.matmul(mat,eigv[:,i])
        for j in range(first_n):
            op_ss[j][i]=numpy.vdot(v_bra,eigv[:,j])
    print(op_ss)
    eig_ss,eigv_ss=numpy.linalg.eig(op_ss)
    print(eig_ss)

def gen_smp_and_ha():
    """generate smapling and hamiltonian"""
    m,n=(2,3)
    fname="2x3.ha"
    #generate samples
    """state_index=[]
    for i in sample(m*n*2):
        if (i==1).sum()%2==0:
            state_index.append(i.tolist())
    state_index=torch.tensor(state_index,dtype=torch.float)"""
    state_index=sample(m*n*2)
    #generate Hamiltonian
    edges=gen_edges((m,n))
    ex=[i[0:2] for i in edges if i[2]%3==X_FLAG]
    ey=[i[0:2] for i in edges if i[2]%3==Y_FLAG]
    ez=[i[0:2] for i in edges if i[2]%3==Z_FLAG]
    ha_list=[[szsz,ez],[sxsx,ex],[sysy,ey]]
    H=gen_H(state_index,ha_list)
    #assert and save
    assert state_index.shape==(4096,m*n*2)
    assert (H==H.transpose(0,1)).sum()==4096**2
    with open(fname,'wb') as f:
        pickle.dump((state_index,H),f)
        log("successfully dumped (state_index,H) to %s"%(fname))

def main():
    fname="2x3_part.ha"
    with open(fname,'rb') as f:
        state_index,H=pickle.load(f)
        log("successfully loaded (state_index,H) from %s"%(fname))

    torch.set_default_dtype(torch.float64)
    state_index=state_index.double()
    H=H.double()
    if len(sys.argv)==2:
        t=RBM(state_index,H,"cuda:%s"%(sys.argv[1]))
    else:
        t=RBM(state_index,H)

    id_np=numpy. identity(state_index.shape[0])
    left_np=gen_ops_npmat([],[],state_index,rots=[(0,2,4),(1,3,5),(6,8,10),(7,9,11)])
    left_two_np=numpy.matmul(left_np,left_np)
    down_np=gen_ops_npmat([],[],state_index,rots=[(0,6),(1,7),(2,8),(3,9),(4,10),(5,11)])
    assert numpy.array_equal(numpy.matmul(down_np,down_np),id_np)
    assert numpy.array_equal(numpy.matmul(numpy.matmul(left_np,left_np),left_np),id_np)
    assert numpy.array_equal(numpy.matmul(down_np,left_np),numpy.matmul(left_np,down_np))
    Sx=numpy.array([[0j,1+0j],[1+0j,0j]])
    Sy=numpy.array([[0j,-1j],[1j,0j]])
    Sz=numpy.array([[1+0j,0j],[0j,-1+0j]])
    plaq_c=gen_ops_npmat([Sy,Sz,Sx,Sy,Sz,Sx],[3,4,5,10,9,8],state_index)
    plaq_b=gen_ops_npmat([Sy,Sz,Sx,Sy,Sz,Sx],[1,2,3,8,7,6],state_index)
    assert numpy.array_equal(numpy.matmul(plaq_c,plaq_b),numpy.matmul(plaq_b,plaq_c))
    #sym_np=(id_np+down_np)/2
    #sym_np=(id_np+left_np+left_two_np)/3
    sym_np=(id_np+down_np+left_np+left_two_np+numpy.matmul(left_np,down_np)+numpy.matmul(left_two_np,down_np))/6
    #sym_np=(id_np+plaq_b)/2
    #sym_np=(id_np+plaq_c+plaq_b+numpy.matmul(plaq_c,plaq_b))/4
    t.symmetry=torch.tensor(sym_np.real,device=t.device,dtype=torch.float64)
    t.train()

if __name__=="__main__":
    #main()
    #gen_smp_and_ha()
    test_gen_ops_npmat()

"""def oth_init(self):
        num_r=10
        w1_re_clone=self.w1_re.weight.clone().detach()
        for i in range(self.hidden_num_2):
            if i>=1:
                w2_re_clone=self.w2_re.weight[0:i,:].clone().detach()
                ws_re=torch.cat((w1_re_clone,w2_re_clone))
            else:
                ws_re=w1_re_clone
            r=(torch.rand(num_r,self.spinor_num,device=self.device)-0.5)*0.5*w1_re_clone.abs().max()
            l=[(i,RBM.dst(j,ws_re)) for j in r]
            l.sort(key=lambda x:x[1],reverse=True)
            with torch.no_grad():
                self.w2_re.weight.data[i,:]=r[l[i][0],:]
        log("inited w2_re: %s"%(self.w2_re.weight.data[-1,:]))

        w1_im_clone=self.w1_im.weight.clone().detach()
        for i in range(self.hidden_num_2):
            if i>=1:
                w2_im_clone=self.w2_im.weight[0:i,:].clone().detach()
                ws_im=torch.cat((w1_im_clone,w2_im_clone))
            else:
                ws_im=w1_im_clone
            r=(torch.rand(num_r,self.spinor_num,device=self.device)-0.5)*0.5*w1_im_clone.abs().max()
            l=[(i,RBM.dst(r[i,:],ws_im)) for i in range(num_r)]
            l.sort(key=lambda x:x[1],reverse=True)
            with torch.no_grad():
                self.w2_im.weight.data[i,:]=r[l[i][0],:]
        log("inited w2_im: %s"%(self.w2_im.weight.data[-1,:]))"""