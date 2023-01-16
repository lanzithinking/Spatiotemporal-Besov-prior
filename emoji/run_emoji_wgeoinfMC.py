"""
Main function to run whitened (geometric) dimension-independent sampling for the dynamic linear inverse problem of emoji.
----------------------
Shiwei Lan @ ASU, 2022
----------------------
"""

# modules
import os,argparse#,pickle
import numpy as np
import timeit,time
from scipy import stats

# the inverse problem
from emoji import emoji
from posterior import *

import sys
sys.path.append( "../" )
# utility
# from util.Eigen import *
# MCMC
from sampler.wht_geoinfMC import wht_geoinfMC

# basic settings
np.set_printoptions(precision=3, suppress=True)
import warnings
warnings.filterwarnings(action="once")

def main(seed=2022):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('alg_NO', nargs='?', type=int, default=3)
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=2000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=(.01,9e-8,9e-8,.001,.001))
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN','winfMALA','winfHMC','winfmMALA','winfmHMC'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    # define emoji Bayesian inverse problem
    spat_args={'basis_opt':'Fourier','l':1,'s':2,'q':1.0,'L':2000}
    # spat_args={'basis_opt':'wavelet','wvlet_typ':'Meyer','l':1,'s':2,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100,'sigma':10}
    store_eig = True
    emj = emoji(spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
    # transformation
    nmlz = lambda z,q=1: z/np.linalg.norm(z,axis=1)[:,None]**q
    def Lmd(z, dord=0, q=emj.prior.qep.q):
        _z = z.reshape((emj.prior.L,emj.prior.qep.N,-1),order='F') # (L,J,_)
        nm_z = np.linalg.norm(_z,axis=1,keepdims=True)
        if dord==0:
            return emj.prior.qep.act(_z*nm_z**(2/q-1),alpha=0.5,transp=True).squeeze()#,chol=False) # (L,J)
        if dord==1:
            def grad(v, adj=False):
                _v = v.reshape((emj.prior.L,emj.prior.qep.N,-1),order='F')
                if adj:
                    _v = emj.prior.qep.act(_v, alpha=0.5,transp=True,adjt=adj)#,chol=False)
                    dLmdv = _z*np.sum(_z*_v,axis=1,keepdims=True)*nm_z**(2/q-3)*(2/q-1) + _v*nm_z**(2/q-1)
                    return dLmdv.squeeze()
                else:
                    return emj.prior.qep.act(_z*np.sum(_z*_v,axis=1,keepdims=True)*nm_z**(2/q-3)*(2/q-1) + _v*nm_z**(2/q-1), alpha=0.5,transp=True).squeeze()#,chol=False)
            return grad
        if dord==2:
            def hess(v, w, adj=False):
                _v = v.reshape((emj.prior.L,emj.prior.qep.N,-1),order='F')
                _w = w.reshape((emj.prior.L,emj.prior.qep.N,-1),order='F')
                Hv0 = (2/q-1)*emj.prior.qep.act(_w*np.sum(_z*_v,axis=1,keepdims=True)*nm_z**(2/q-3), alpha=0.5,transp=True,adjt=adj)
                Hv1 = (2/q-1)*emj.prior.qep.act(_z*nm_z**(2/q-3), alpha=0.5,transp=True)
                Hv2 = (2/q-1)*emj.prior.qep.act(_z*np.sum(_z*_v,axis=1,keepdims=True)*nm_z**(2/q-5)*(2/q-3) + _v*nm_z**(2/q-3), alpha=0.5,transp=True)
                if adj:
                    wHv = Hv0 + np.sum(_w*Hv1,axis=1,keepdims=True)*_v + np.sum(_w*Hv2,axis=1,keepdims=True)*_z
                else:
                    wHv = Hv0 + np.sum(_w*_v,axis=1,keepdims=True)*Hv1 + np.sum(_w*_z,axis=1,keepdims=True)*Hv2
                return wHv.squeeze()
            return hess
    # h=1e-8; z, v, w=np.random.randn(3,emj.prior.bsv.L*emj.prior.qep.N)
    # val,grad,hess=Lmd(z,0),Lmd(z,1),Lmd(z,2)
    # val1,grad1=Lmd(z+h*v,0),Lmd(z+h*w,1)
    # print('error in gradient: %0.8f' %(np.linalg.norm((val1-val)/h-grad(v))/np.linalg.norm(v)))
    # print('error in Hessian: %0.8f' %(np.linalg.norm((grad1(v)-grad(v))/h-hess(v,w))/np.sqrt(np.linalg.norm(v)*np.linalg.norm(w))))
    def T(z, dord=0, q=emj.prior.bsv.q):
        if dord==0:
            return emj.prior.C_act(Lmd(z, dord), 1/q).squeeze()
        if dord==1:
            return lambda v,adj=False: Lmd(z, dord)(emj.prior.C_act(v, 1/q),adj=adj).reshape(v.shape,order='F') if adj else emj.prior.C_act(Lmd(z, dord)(v), 1/q).squeeze()
        if dord==2:
            return lambda v,w,adj=False: Lmd(z, dord)(emj.prior.C_act(v, 1/q), w,adj=adj).reshape(v.shape,order='F')
    # h=1e-8; z, v, w=np.random.randn(3,emj.prior.bsv.L*emj.prior.qep.N)
    # val,grad,hess=T(z,0),T(z,1),T(z,2)
    # val1,grad1=T(z+h*v,0),T(z+h*w,1)
    # print('error in gradient: %0.8f' %(np.linalg.norm((val1-val)/h-grad(v))/np.linalg.norm(v)))
    # print('error in Hessian: %0.8f' %(np.linalg.norm((grad1(v)-grad(v))/h-hess(v,w))/np.sqrt(np.linalg.norm(v)*np.linalg.norm(w))))
    invLmd = lambda xi,q=emj.prior.qep.q: nmlz(emj.prior.qep.act(xi.reshape((-1,emj.prior.qep.N),order='F'),alpha=-0.5,transp=True),1-q/2)
    invT = lambda u,q=emj.prior.bsv.q: invLmd(emj.prior.C_act(u, -1/q))
    # transformed geometry of white noise parameter
    def geomT(parameter=None,geom_ord=[0],**kwargs):
        MF_only=kwargs.pop('MF_only',True)
        l=None; g=None; invK_=None; Keigs=None;
        param = T(parameter) if callable(T) else parameter
        # l,g_,invK_,Keigs = emj.get_geom(param, geom_ord, MF_only=MF_only, adjust_grad=True, store_eig=False, L=100)
        if 0 in geom_ord: l=-emj._get_misfit(param)
        if 1 in geom_ord: g=-emj._get_grad(param)
        if 2 in geom_ord:
            invK_=emj._get_HessApply(param,MF_only=MF_only)
            g+=invK_(param)
            if not MF_only: g-=param
        if g is not None: g = T(parameter, 1)(g, adj=True)
        invK = invK_ if invK_ is None else lambda v: T(parameter, 1)(invK_(v), adj=True) + T(parameter, 2)(v, emj._get_grad(param, MF_only=MF_only), adj=True)
        # emj.posterior = posterior(invK=spsla.LinearOperator((parameter.size,)*2,invK),L=50,store_eig=True)
        emj.posterior = posterior(invK, N=parameter.size,L=50,store_eig=True)
        Keigs = emj.posterior.eigs(**kwargs)
        return l,g,invK,Keigs
    
    # initialization random noise epsilon
    try:
        # fld=os.path.join(os.getcwd(),'reconstruction/MAP')
        # with open(os.path.join(fld,'MAP.pckl'),'rb') as f:
        #     map=pickle.load(f)
        # f.close()
        # u=invT(map).flatten(order='F')
        u_init=emj.init_parameter if hasattr(emj,'init_parameter') else emj._init_param(init_opt='LSE',lmda=10)
        u=invT(u_init).flatten(order='F')
    except Exception as e:
        print(e)
        u=np.random.randn({'vec':emj.prior.L*emj.prior.qep.N,'fun':emj.prior.N}[emj.prior.space])
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s) usign random seed %d..."
          % (args.algs[args.alg_NO],args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO],args.seed_NO))
    
    winfMC=wht_geoinfMC(u_init,emj,args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO],args.algs[args.alg_NO],transformation=T,geomT=geomT, MF_only=False)
    res=winfMC.sample(args.num_samp,args.num_burnin,return_result=True)
    
    # samp=[]; loglik=[]; times=[]
    # accp=0; acpt=0
    # sampler=getattr(winfMC,args.algs[args.alg_NO])
    # prog=np.ceil((args.num_samp+args.num_burnin)*(.05+np.arange(0,1,.05)))
    # beginning=timeit.default_timer()
    # for i in range(args.num_samp+args.num_burnin):
    #     if i==args.num_burnin:
    #         # start the timer
    #         tic=timeit.default_timer()
    #         print('\nBurn-in completed; recording samples now...\n')
    #     # generate MCMC sample with given sampler
    #     acpt_ind,_=sampler()
    #     u,l=winfMC.u,winfMC.ll
    #     # display acceptance at intervals
    #     if i+1 in prog:
    #         print('{0:.0f}% has been completed.'.format(np.float(i+1)/(args.num_samp+args.num_burnin)*100))
    #     # online acceptance rate
    #     accp+=acpt_ind
    #     if (i+1)%100==0:
    #         print('Acceptance at %d iterations: %0.2f' % (i+1,accp/100))
    #         accp=0.0
    #     # save results
    #     loglik.append(l)
    #     if i>=args.num_burnin:
    #         samp.append(T(u))
    #         acpt+=acpt_ind
    #     times.append(timeit.default_timer()-beginning)
    # # stop timer
    # toc=timeit.default_timer()
    # time_=toc-tic
    # acpt/=args.num_samp
    # print("\nAfter %g seconds, %d samples have been collected. \n" % (time_,args.num_samp))
    #
    # # store the results
    # samp=np.stack(samp); loglik=np.stack(loglik);  times=np.stack(times)
    # # name file
    # ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # savepath=os.path.join(os.getcwd(),'result')
    # if not os.path.exists(savepath): os.makedirs(savepath)
    # filename='emj_'+args.algs[args.alg_NO]+'_dim'+str(len(u))+'_'+ctime
    # np.savez_compressed(os.path.join(savepath,filename),spat_args=spat_args, temp_args=temp_args, args=args, samp=samp,loglik=loglik,time_=time_,times=times)
    
    # plot
    # loaded=np.load(os.path.join(savepath,filename+'.npz'))
    # samp=loaded['samp']
    samp=res[3]
    
    mcmc_v_med = np.median(samp,axis=0)
    mcmc_v_mean = np.mean(samp,axis=0)
    mcmc_v_std = np.std(samp,axis=0)
    mcmc_f = emj.prior.vec2fun(mcmc_v_med).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    emj.misfit.plot_reconstruction(rcstr_imgs=mcmc_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_median')
    mcmc_f = emj.prior.vec2fun(mcmc_v_mean).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    emj.misfit.plot_reconstruction(rcstr_imgs=mcmc_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_mean')
    mcmc_f = emj.prior.vec2fun(mcmc_v_std).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    emj.misfit.plot_reconstruction(rcstr_imgs=mcmc_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_std')

if __name__ == '__main__':
    main()
    # n_seed = 10; i=0; n_success=0
    # while n_success < n_seed:
    #     seed_i=2022+i*10
    #     try:
    #         print("Running for seed %d ...\n"% (seed_i))
    #         main(seed=seed_i)
    #         n_success+=1
    #     except Exception as e:
    #         print(e)
    #         pass
    #     i+=1