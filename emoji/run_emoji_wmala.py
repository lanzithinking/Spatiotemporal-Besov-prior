"""
Main function to run whitened preconditioned Crank-Nicolson sampling for the time series problem
----------------------
Shiwei Lan @ ASU, 2022
----------------------
created by Shuyi Li
"""

# modules
import os,argparse
import numpy as np
import timeit,time
from scipy import stats
import scipy.sparse as sps

# the inverse problem
from emoji import emoji

# MCMC
import sys
sys.path.append( "../" )
from sampler.wMALA import wMALA


np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=1)
    parser.add_argument('num_burnin', nargs='?', type=int, default=0)    
    parser.add_argument('alg_NO', nargs='?', type=int, default=0)
    parser.add_argument('step_sizes', nargs='?', type=float, default=(.01,))
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN',))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    # define emoji Bayesian inverse problem
    spat_args={'basis_opt':'Fourier','l':1,'s':2,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    store_eig = True
    emj = emoji(spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed, init_param=True)
    logLik = lambda u: -emj._get_misfit(u, MF_only=True)
    # transformation
    nmlz = lambda z,q=1: z/np.linalg.norm(z,axis=1)[:,None]**q
    Lmd = lambda z,q=emj.prior.qep.q: emj.prior.qep.act(nmlz(z.reshape((-1,emj.prior.qep.N),order='F'),1-2/q),alpha=0.5,transp=True)
    T = lambda z,q=emj.prior.bsv.q: emj.prior.C_act(Lmd(z), 1/q)
    invLmd = lambda xi,q=emj.prior.qep.q: nmlz(emj.prior.qep.act(xi.reshape((-1,emj.prior.qep.N),order='F'),alpha=-0.5,transp=True),1-q/2)
    invT = lambda u,q=emj.prior.bsv.q: invLmd(emj.prior.C_act(u, -1/q))
    
    eigv, _=emj.prior.bsv.eigs()
    #D_lmd = dT(lmd)/d lmd [prior.tomat**1/q] * d(misfit)/d(T(lmd))
    D_lmd = lambda x,q=emj.prior.bsv.q: sps.diags(np.tile(eigv**(1/q), emj.prior.J)).dot(emj._get_grad(T(x)))
    
    # initialization random noise epsilon
    try:
        # fld=os.path.join(os.getcwd(),'reconstruction/MAP')
        # with open(os.path.join(fld,'MAP.pckl'),'rb') as f:
        #     map=pickle.load(f)
        # f.close()
        # u=invT(map).flatten(order='F')
        u_init=emj.init_parameter if hasattr(emj,'init_parameter') else emj._init_param()
        u=invT(u_init).flatten(order='F')
    except Exception as e:
        print(e)
        u=np.random.randn({'vec':emj.prior.L*emj.prior.qep.N,'fun':emj.prior.N}[emj.prior.space])
    l=logLik(T(u))
    # current energy I(u0, u)
    h = args.step_sizes[args.alg_NO]
    beta = 4*np.sqrt(h)/(4+h)
    E_cur = l + h/8*D_lmd(u).dot(D_lmd(u)) - np.sqrt(h)/2*D_lmd(u).dot((u - np.sqrt(1-beta**2)*u)/beta) 
    
    # run MCMC to generate samples
    print("Running the whitened preconditioned MALA (wMALA) for %s prior model taking random seed %d ..." % ('Besov', args.seed_NO))
    
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
        
        u,l,E_cur,acpt_ind=wMALA(u,l,E_cur,D_lmd,logLik,T,h=h)
        # display acceptance at intervals
        if i+1 in prog:
            print('{0:.0f}% has been completed.'.format(np.float(i+1)/(args.num_samp+args.num_burnin)*100))
        # online acceptance rate
        accp+=acpt_ind
        if (i+1)%100==0:
            print('Acceptance at %d iterations: %0.2f' % (i+1,accp/100))
            accp=0.0
        # save results
        loglik.append(l)
        if i>=args.num_burnin:
            samp.append(T(u))
            acpt+=acpt_ind
        times.append(timeit.default_timer()-beginning)
    # stop timer
    toc=timeit.default_timer()
    time_=toc-tic
    acpt/=args.num_samp
    print("\nAfter %g seconds, %d samples have been collected. \n" % (time_,args.num_samp))
    
    # store the results
    samp=np.stack(samp); loglik=np.stack(loglik);  times=np.stack(times)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'result')
    if not os.path.exists(savepath): os.makedirs(savepath)
    filename='emj_wpCN_dim'+str(len(u))+'_'+ctime
    np.savez_compressed(os.path.join(savepath,filename),spat_args=spat_args, temp_args=temp_args, args=args, samp=samp,loglik=loglik,time_=time_,times=times)
    
    # plot
    # loaded=np.load(os.path.join(savepath,filename+'.npz'))
    # samp=loaded['samp']
    
    mcmc_v_med = np.median(samp,axis=0)
    mcmc_v_mean = np.mean(samp,axis=0)
    mcmc_f = emj.prior.vec2fun(mcmc_v_med).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    emj.misfit.plot_reconstruction(rcstr_imgs=mcmc_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_median')
    mcmc_f = emj.prior.vec2fun(mcmc_v_mean).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    emj.misfit.plot_reconstruction(rcstr_imgs=mcmc_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_mean')


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