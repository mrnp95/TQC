#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import netket,numpy,time,json,sys,os,itertools

sz=[[1,0],[0,-1]]
sx=[[0,1],[1,0]]
sy=[[0,-1j],[1j,0]]
def show_graph(g):
    print("graph spinor number: %d, edge number: %d, is_bipartite: %s"%(g.n_sites,len(g.edges),g.is_bipartite))
def show_hilbert(h):
    try:
        print("hilbert size: %d, n_states: %d"%(h.size,h.n_states))
    except Exception as e:
        print(e)
        print("hilbert size: %d"%(h.size))
def show_machine(m):
    print("machine.parameters:\n%s"%(machine.parameters))
def get_hamiltonian(L,show=True):
    edges=[[i,(i+1)%L,0] for i in range(L)]
    graph=netket.graph.CustomGraph(edges)
    hilbert=netket.hilbert.Spin(s=0.5,graph=graph)
    hamiltonian=netket.operator.GraphOperator(hilbert
                ,bondops=[(-1*numpy.kron(sz,sz)).tolist()]
                ,bondops_colors=[0])
    if show:
        show_graph(graph)
        show_hilbert(hilbert)
    return hamiltonian
def measure(hamiltonian,machine,observables,output_prefix=None,delete_temp=True,total_samples=1000000,batch_samples=100):
    sa=netket.sampler.MetropolisLocal(machine=machine)
    #op=netket.optimizer.Sgd(learning_rate=0,decay_factor=0)
    op=netket.optimizer.Momentum(learning_rate=0,beta=0)
    gs=netket.variational.Vmc(hamiltonian=hamiltonian,sampler=sa,optimizer=op,n_samples=batch_samples,diag_shift=0.1)
    for k in observables:
        gs.add_observable(observables[k],str(k))
    if output_prefix==None:
        output_prefix="%f.measuretemp"%(time.time(),)
    elif output_prefix.endswith(".measuretemp"):
        output_prefix+=".measuretemp"
    print("begin measure %d obervables with %d:%d"%(len(observables),total_samples,batch_samples))
    start=time.time()
    gs.run(output_prefix=output_prefix,n_iter=int(total_samples/batch_samples))
    end=time.time()
    print('measuring %d observables takes %fs'%(len(observables),end-start))
    data=json.load(open(output_prefix+".log"))
    data_output=data["Output"]
    results={}
    for k in list(observables)+["Energy"]:
        results[k]={}
        results[k]["Mean"]=[i[k]["Mean"] for i in data_output]
        results[k]["Sigma"]=[i[k]["Sigma"] for i in data_output]
        results[k]["Taucorr"]=[i[k]["Taucorr"] for i in data_output]
        #print(results[k]["Mean"])
        try:
            results[k]["mean_Mean"]=numpy.mean(results[k]["Mean"])
            #results[k]["std_Mean"]=numpy.std(results[k]["Mean"])
            results[k]["mean_Sigma"]=numpy.mean(results[k]["Sigma"])
            #results[k]["std_Sigma"]=numpy.std(results[k]["Sigma"])
        except Exception as e:
            print("failed to measure %s: %s"%(k,e))
            results[k]["mean_Mean"]=0
            #results[k]["std_Mean"]=0
            results[k]["mean_Sigma"]=0
            #results[k]["std_Sigma"]=0
        results[k]["Error"]=results[k]["mean_Sigma"]/numpy.sqrt(len(results[k]["Sigma"]))
    if delete_temp:
        if os.system("rm %s*"%(output_prefix))!=0:
            print("remove tempdata failed")
        else:
            print("temp file %s deleted"%(output_prefix))
    for k in results:
        #print("%s:mean_Mean=%9.6f,std_Mean=%f,mean_Sigma=%f,std_Sigma=%f"\
        #print("%s|%6.3f|%.4f|%.4f|%.4f|"\
        #        %(k,results[k]["mean_Mean"],results[k]["std_Mean"],results[k]["mean_Sigma"],results[k]["std_Sigma"]))
        #print("%s|%.4f(%7.4f)|"%(k,results[k]["mean_Sigma"],results[k]["mean_Mean"],))
        print("%s: %f(%f)"%(k,results[k]["mean_Mean"],results[k]["Error"]))
    return results
def exact_measure(parameters,L,n_h=None):
    if n_h==None:
        n_h=L
    a=numpy.matrix(parameters[0:L]).T
    b=numpy.matrix(parameters[L:L+n_h]).T
    W=numpy.matrix(parameters[L+n_h:L+n_h+L*(n_h)]).reshape(n_h,L)
    #print("a: %s\nb: %s\nW: %s"%(a,b,W))
    l_states=[];Psi=[]
    for i in itertools.product([1,-1],repeat=L):
        l_states.append(i)
        i=numpy.matrix(i).T
        psi=numpy.exp(i.T*a)
        for j in numpy.cosh(W*i+b):
            psi*=j
        Psi.append(psi.tolist()[0][0])
    print("l_states: %s\nPsi: %s"%(l_states,Psi))
    sz0=[]
    sx0=[]
    sy0=[]
    Weights=[]
    msx=numpy.matrix(sx)
    msy=numpy.matrix(sy)
    msz=numpy.matrix(sz)
    up=numpy.matrix([[1],[0]])
    down=numpy.matrix([[0],[1]])
    for i,j in itertools.product(range(len(l_states)),repeat=2):
        Weights.append(Psi[i]*Psi[j].conjugate())
        if l_states[i][0]==1:
            statei=up
        else:
            statei=down
        if l_states[j][0]==1:
            statej=up
        else:
            statej=down
        print(statej.T.conjugate()*msx*statei*Weights[-1])
    print(numpy.sum(Weights))
    print(numpy.sum(sx0)/numpy.sum(Weights))
if __name__=="__main__":
    L=3
    hamiltonian=get_hamiltonian(L)
    machine=netket.machine.RbmSpin(hilbert=hamiltonian.hilbert,n_hidden=L)
    machine.load("good3s3n.madata")
    #machine.init_random_parameters(sigma=0.1,seed=int(time.time()*1000000)%65535)
    #machine.save("test.madata")
    show_machine(machine)
    exact_measure(machine.parameters,L)
    raise Exception
    sz0=netket.operator.LocalOperator(hamiltonian.hilbert,[sz,],[[0],])
    sy0=netket.operator.LocalOperator(hamiltonian.hilbert,[sy,],[[0],])
    sx0=netket.operator.LocalOperator(hamiltonian.hilbert,[sx,],[[0],])
    measure(hamiltonian,machine,{"sz0":sz0,"sy0":sy0,"sx0":sx0},delete_temp=False,total_samples=1000000,batch_samples=1000)
    #show_machine(machine)