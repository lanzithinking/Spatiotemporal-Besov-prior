"""
Main function to run whitened (geometric) dimension-independent sampling for the dynamic linear inverse problem of STEMPO.
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
from STEMPO import STEMPO

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
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=2000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=(1e-7,1e-6,1e-6,1e-3,1e-3))
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN','winfMALA','winfHMC','winfmMALA','winfmHMC'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    # define STEMPO Bayesian inverse problem
    data_args={'data_set':'simulation'}
    spat_args={'basis_opt':'Fourier','l':1,'s':1,'q':1.0,'L':2000}
    # spat_args={'basis_opt':'wavelet','wvlet_typ':'Meyer','l':1,'s':2,'q':1.0,'L':2000}
    # temp_args={'ker_opt':'powexp','l':.5,'s':2,'q':1.0,'L':100}
    temp_args={'ker_opt':'matern','l':.5,'s':2,'q':1.0,'L':100}
    store_eig = True
    stpo = STEMPO(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
    
    # initialization random noise epsilon
    try:
        # fld=os.path.join(os.getcwd(),'reconstruction/MAP')
        # with open(os.path.join(fld,'MAP.pckl'),'rb') as f:
        #     map=pickle.load(f)
        # f.close()
        # z_init=stpo.whiten.stbp2wn(map).flatten(order='F')
        u_init=stpo.init_parameter if hasattr(stpo,'init_parameter') else stpo._init_param(init_opt='LSE',lmda=10)
        z_init=stpo.whiten.stbp2wn(u_init).flatten(order='F')
    except Exception as e:
        print(e)
        z_init=np.random.randn({'vec':stpo.prior.L*stpo.prior.J,'fun':stpo.prior.N}[stpo.prior.space])
    # h=1e-7; v=np.random.randn(stpo.prior.L*stpo.prior.J)
    # l,g=stpo.get_geom(z_init,geom_ord=[0,1],whiten=True)[:2]; hess=stpo.get_geom(z_init,geom_ord=[2],whiten=True)[2]
    # Hv=hess(v)
    # l1,g1=stpo.get_geom(z_init+h*v,geom_ord=[0,1],whiten=True)[:2]
    # print('error in gradient: %0.8f' %(abs((l1-l)/h-g.dot(v))/np.linalg.norm(v)))
    # print('error in Hessian: %0.8f' %(np.linalg.norm(-(g1-g)/h-Hv)/np.linalg.norm(v)))
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s) using random seed %d..."
          % (args.algs[args.alg_NO],args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO],args.seed_NO))
    
    winfMC=wht_geoinfMC(z_init,stpo,args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO],args.algs[args.alg_NO],transformation=stpo.whiten.wn2stbp, MF_only=True, whitened=True, k=100)
    res=winfMC.sample(args.num_samp,args.num_burnin,return_result=True)#, save_result=False)
    
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
    # filename='stpo_'+args.algs[args.alg_NO]+'_dim'+str(len(u))+'_'+ctime
    # np.savez_compressed(os.path.join(savepath,filename),spat_args=spat_args, temp_args=temp_args, args=args, samp=samp,loglik=loglik,time_=time_,times=times)
    
    # plot
    # loaded=np.load(os.path.join(savepath,filename+'.npz'))
    # samp=loaded['samp']
    samp=res[3]
    
    try:
        if stpo.prior.space=='vec': samp=stpo.prior.vec2fun(samp.T).T
        med_f = np.median(samp,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F').swapaxes(0,1)
        mean_f = np.mean(samp,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F').swapaxes(0,1)
        std_f = np.std(samp,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F').swapaxes(0,1)
    except Exception as e:
        print(e)
        mean_f=0; std_f=0
        n_samp=samp.shape[0]
        for i in range(n_samp):
            samp_i=stpo.prior.vec2fun(samp[i]) if stpo.prior.space=='vec' else samp[i]
            mean_f+=samp_i/n_samp
            std_f+=samp_i**2/n_samp
        std_f=np.sqrt(std_f-mean_f**2)
        mean_f=mean_f.reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F').swapaxes(0,1)
        std_f=std_f.reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F').swapaxes(0,1)
        med_f=None
    if med_f is not None:
        stpo.misfit.plot_reconstruction(rcstr_imgs=med_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_median')
    stpo.misfit.plot_reconstruction(rcstr_imgs=mean_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_mean')
    stpo.misfit.plot_reconstruction(rcstr_imgs=std_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_std')

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