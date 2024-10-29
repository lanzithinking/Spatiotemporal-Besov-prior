"""
Main function to run whitened (geometric) dimension-independent sampling for the example of NOAA temperature.
----------------------
Shiwei Lan @ ASU, 2022
----------------------
"""

# modules
import os,argparse,pickle
import numpy as np
import timeit,time
from scipy import stats

# the inverse problem
from NOAATMP import NOAATMP

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
    parser.add_argument('alg_NO', nargs='?', type=int, default=1)
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=2000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=(1e-6, 5e-7,1e-5,1e-3,1e-3)) # q1: 2e-8, q2 & iidT: 5e-7
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN','winfMALA','winfHMC','winfmMALA','winfmHMC'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    # define emoji Bayesian inverse problem
    data_args={'impute_missing':False,'nzlvl':.1,'holdout_percent':0.1,'random_seed':seed}
    spat_args={'basis_opt':'Fourier','sigma2':100,'l':.1,'s':1,'q':args.q,'L':2000}
    temp_args={'ker_opt':'matern','sigma2':10,'l':1e-10,'s':1,'q':args.q,'L':100}
    store_eig = True
    noaatmp = NOAATMP(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
    if noaatmp.misfit.hldt_pcnt>0:
        truth = noaatmp.misfit.obs[noaatmp.misfit.holdout_msk]
    else:
        truth = noaatmp.misfit._impute(axis=[0,1])
    
    # initialization random noise epsilon
    try:
        # fld=os.path.join(os.getcwd(),'properties')
        # with open(os.path.join(fld,'MAP.pckl'),'rb') as f:
        #     map=pickle.load(f)
        # f.close()
        # z_init=noaatmp.whiten.stbp2wn(map).flatten(order='F')
        u_init=noaatmp.init_parameter if hasattr(noaatmp,'init_parameter') else noaatmp._init_param(init_opt='LSE', axis=[0,1])
        z_init=noaatmp.whiten.stbp2wn(u_init).flatten(order='F')
    except Exception as e:
        print(e)
        z_init=noaatmp.whiten.sample()
    # h=1e-7; v=np.random.randn(noaatmp.prior.L*noaatmp.prior.J)
    # l,g=noaatmp.get_geom(z_init,geom_ord=[0,1],whitened=True)[:2]; hess=noaatmp.get_geom(z_init,geom_ord=[1.5],whitened=True)[2]
    # Hv=hess(v)
    # l1,g1=noaatmp.get_geom(z_init+h*v,geom_ord=[0,1],whitened=True)[:2]
    # print('error in gradient: %0.8f' %(abs((l1-l)/h-g.dot(v))/np.linalg.norm(v)))
    # print('error in Hessian: %0.8f' %(np.linalg.norm(-(g1-g)/h-Hv)/np.linalg.norm(v)))
    
    # # center priors
    # noaatmp.prior.mean = u_init
    # noaatmp.whiten.mean = noaatmp.whiten.stbp2wn(noaatmp.prior.mean)
    
    # adjust the sample size
    if args.alg_NO>=3:
        args.num_samp=2000
        args.num_burnin=1000
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s) using random seed %d..."
          % (args.algs[args.alg_NO],args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO],args.seed_NO))
    
    winfMC=wht_geoinfMC(z_init,noaatmp,args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO],args.algs[args.alg_NO],transformation=noaatmp.whiten.wn2stbp, MF_only=True, whitened=True, k=100)
    res=winfMC.sample(args.num_samp,args.num_burnin,return_result=True, save_result=False)
    
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
    # filename='noaatmp_'+args.algs[args.alg_NO]+'_dim'+str(len(u))+'_'+ctime
    # np.savez_compressed(os.path.join(savepath,filename),spat_args=spat_args, temp_args=temp_args, args=args, samp=samp,loglik=loglik,time_=time_,times=times)
    
    # plot
    # loaded=np.load(os.path.join(savepath,filename+'.npz'))
    # samp=loaded['samp']
    samp=res[3]
    
    try:
        if noaatmp.prior.space=='vec': samp=noaatmp.prior.vec2fun(samp.T).T
        med_f = np.median(samp,axis=0).reshape(noaatmp.misfit.obs.shape, order='F')
        mean_f = np.mean(samp,axis=0).reshape(noaatmp.misfit.obs.shape, order='F')
        std_f = np.std(samp,axis=0).reshape(noaatmp.misfit.obs.shape, order='F')
    except Exception as e:
        print(e)
        mean_f=0; std_f=0; med_f=None
        n_samp=samp.shape[0]
        for i in range(n_samp):
            samp_i=noaatmp.prior.vec2fun(samp[i]) if noaatmp.prior.space=='vec' else samp[i]
            mean_f+=samp_i/n_samp
            std_f+=samp_i**2/n_samp
        std_f=np.sqrt(std_f-mean_f**2)
        mean_f=mean_f.reshape(noaatmp.misfit.obs.shape, order='F')
        std_f=std_f.reshape(noaatmp.misfit.obs.shape, order='F')
    f_name=spat_args['basis_opt']+('_'+spat_args['wvlet_typ'] if spat_args['basis_opt']=='wavelet' else '')+'_'+temp_args['ker_opt']\
           +('_iidT' if temp_args['l']<1e-5 else '_q'+str(args.q))+'_hldt'+str(noaatmp.misfit.hldt_pcnt)
    if med_f is not None:
        noaatmp.misfit.plot_reconstruction(rcstr_imgs=med_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_'+f_name+'_median', snap_idx=np.linspace(0,noaatmp.misfit.sz_t,num=10,endpoint=False).astype(int), label=False)
    noaatmp.misfit.plot_reconstruction(rcstr_imgs=mean_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_'+f_name+'_mean', snap_idx=np.linspace(0,noaatmp.misfit.sz_t,num=10,endpoint=False).astype(int), label=False)
    noaatmp.misfit.plot_reconstruction(rcstr_imgs=std_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_'+f_name+'_std', snap_idx=np.linspace(0,noaatmp.misfit.sz_t,num=10,endpoint=False).astype(int), label=False)
    
    # summary
    est_f = med_f if med_f is not None else mean_f
    est2cmp = est_f[noaatmp.misfit.holdout_msk] if noaatmp.misfit.hldt_pcnt>0 else est_f
    relerr = np.linalg.norm(est2cmp -truth)/np.linalg.norm(truth)
    print('Relative error of MAP compared with the truth %.2f%%' % (relerr*100))
    if noaatmp.misfit.hldt_pcnt>0:
        mean_f = mean_f[noaatmp.misfit.holdout_msk]
        std_f = std_f[noaatmp.misfit.holdout_msk]
    PIcvr = np.logical_and(mean_f-1.96*std_f<truth, truth<mean_f+1.96*std_f).mean()
    print('Truth covering rate of prediction interval %.2f%%' % (PIcvr*100))
    # save to text
    os.makedirs('./results', exist_ok=True)
    stats = np.array([relerr, PIcvr, -res[4].mean(), res[-2]])
    stats = np.array([seed,('_iidT' if temp_args['l']<1e-5 else '_q'+str(args.q))]+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'RLE', 'PIC', 'NLL', 'time']
    with open(os.path.join('./results','noaatmp_'+args.algs[args.alg_NO]+'_dim'+str(est_f.size)+'_hldt'+str(noaatmp.misfit.hldt_pcnt)+'.txt'),'ab') as f:
        np.savetxt(f,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2022 else '')
    
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