"""
Main function to run elliptic slice sampling for the dynamic linear inverse problem of emoji.
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
from scipy import stats
import timeit,time

# the inverse problem
from emoji import emoji

# MCMC
import sys
sys.path.append( "../" )
from sampler.ESS import ESS


np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=2000)
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    # define emoji Bayesian inverse problem
    data_args={'data_set':'60proj','data_thinning':2}
    spat_args={'basis_opt':'Fourier','l':1,'s':1,'q':1.0,'L':2000}
    # spat_args={'basis_opt':'wavelet','wvlet_typ':'Meyer','l':1,'s':2,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    store_eig = True
    emj = emoji(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed, init_param=True)
    logLik = lambda u: -emj._get_misfit(u, MF_only=True)
    rnd_pri = emj.prior.sample
    
    # initialization random noise epsilon
    try:
        # fld=os.path.join(os.getcwd(),'reconstruction/MAP')
        # with open(os.path.join(fld,'MAP.pckl'),'rb') as f:
        #     map=pickle.load(f)
        # f.close()
        # u=invT(map).flatten(order='F')
        u=emj.init_parameter if hasattr(emj,'init_parameter') else emj._init_param()
    except Exception as e:
        print(e)
        u=rnd_pri()
    l=logLik(u)
    
    # run MCMC to generate samples
    print("Running the elliptic slice sampler (ESS) for %s prior model taking random seed %d ..." % ('Besov', args.seed_NO))
    
    samp=[]; loglik=[]; times=[]
    accp=0; acpt=0
    prog=np.ceil((args.num_samp+args.num_burnin)*(.05+np.arange(0,1,.05)))
    beginning=timeit.default_timer()
    for i in range(args.num_samp+args.num_burnin):
        if i==args.num_burnin:
            # start the timer
            tic=timeit.default_timer()
            print('\nBurn-in completed; recording samples now...\n')
        # generate MCMC sample with given sampler
        u,l=ESS(u,l,rnd_pri,logLik)
        # display acceptance at intervals
        if i+1 in prog:
            print('{0:.0f}% has been completed.'.format(np.float(i+1)/(args.num_samp+args.num_burnin)*100))
        # save results
        loglik.append(l)
        if i>=args.num_burnin: samp.append(u)
        times.append(timeit.default_timer()-beginning)
    # stop timer
    toc=timeit.default_timer()
    time_=toc-tic
    print("\nAfter %g seconds, %d samples have been collected. \n" % (time_,args.num_samp))
    
    # store the results
    samp=np.stack(samp); loglik=np.stack(loglik);  times=np.stack(times)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'result')
    if not os.path.exists(savepath): os.makedirs(savepath)
    filename='emj_ESS_dim'+str(len(u))+'_'+ctime
    np.savez_compressed(os.path.join(savepath,filename),spat_args=spat_args, temp_args=temp_args, args=args, samp=samp,loglik=loglik,time_=time_,times=times)
    
    # plot
    # loaded=np.load(os.path.join(savepath,filename+'.npz'))
    # samp=loaded['samp']
    
    if emj.prior.space=='vec': samp=emj.prior.vec2fun(samp.T).T
    med_f = np.median(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    mean_f = np.mean(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    std_f = np.std(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    emj.misfit.plot_reconstruction(rcstr_imgs=med_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_median')
    emj.misfit.plot_reconstruction(rcstr_imgs=mean_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_mean')
    emj.misfit.plot_reconstruction(rcstr_imgs=std_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_std')

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