"""
Main function to run elliptic slice sampling for the dynamic linear inverse problem of STEMPO.
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
from scipy import stats
import timeit,time
from scipy import optimize

# the inverse problem
from STEMPO import STEMPO

# MCMC
import sys
sys.path.append( "../" )
from sampler.ESS import ESS
from sampler.slice import slice


np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=2000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    # define STEMPO Bayesian inverse problem
    data_args={'data_set':'simulation'}
    spat_args={'basis_opt':'Fourier','l':.1,'s':1.5,'q':args.q,'L':2000}
    # spat_args={'basis_opt':'wavelet','wvlet_typ':'Meyer','l':1,'s':2,'q':args.q,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    store_eig = True
    stpo = STEMPO(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed, init_param=True)
    logLik = lambda u: -stpo._get_misfit(u, MF_only=True)
    rnd_pri = stpo.prior.sample
    
    # initialization random noise epsilon
    try:
        # fld=os.path.join(os.getcwd(),'reconstruction/MAP')
        # with open(os.path.join(fld,'MAP.pckl'),'rb') as f:
        #     map=pickle.load(f)
        # f.close()
        # u=invT(map).flatten(order='F')
        u=stpo.init_parameter if hasattr(stpo,'init_parameter') else stpo._init_param(init_opt='LSE',lmda=10)
    except Exception as e:
        print(e)
        u=rnd_pri()
    # l=logLik(u)
    
    # specify (hyper)-priors
    # (a,b) in inv-gamma priors for sigma2
    # (m,V) in (log) normal priors for eta_*, (eta=log-l), * = x, t
    a,b=1,1
    m_s,V_s=0,1
    m_t,V_t=0,1
    
    # sigma2 = stats.invgamma.rvs(a, scale=b)**(2/stpo.prior.qep.q)
    sigma2 = stpo.prior.qep.sigma2
    eta_s, eta_t = stats.norm.rvs(m_s,np.sqrt(V_s)), stats.norm.rvs(m_t,np.sqrt(V_t))
    stpo.prior = stpo.prior.update(qep = stpo.prior.qep.update(sigma2=sigma2, l=np.exp(eta_t)),
                                   bsv = stpo.prior.bsv.update(l=np.exp(eta_s)) )
    
    # pre-optimize to find a good start point
    pre_optim_steps = 10
    pre_optim = pre_optim_steps>0
    options={'maxiter':10,'disp':False}
    if pre_optim:
        print("Pre-optimizing hyper-parameters to get a good initial point for MCMC...")
        print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}   {6:9s}   {7:9s}'.format('Iter', ' f(u)', ' sigma2', ' f(sigma2)', ' eta_t', ' f(eta_t)', ' eta_s', 'f(eta_s)'))
    else:
        # stpo.prior.mean=u # center the prior
        print("Running the elliptic slice sampler (ESS) for %s prior model taking random seed %d ..." % ('Besov', args.seed_NO))
    
    # run MCMC to generate samples
    samp_u=[]; loglik=[]; times=[]
    samp_sigma2 = np.zeros((args.num_samp+args.num_burnin))
    samp_eta = np.zeros((args.num_samp+args.num_burnin,2))
    
    prog=np.ceil((args.num_samp+args.num_burnin)*(.05+np.arange(0,1,.05)))
    beginning=timeit.default_timer()
    for i in range(args.num_samp+args.num_burnin):
        if i==pre_optim_steps and pre_optim:
            pre_optim=False; i==0
            print("After %g seconds, initial hyper-parameters %g (sigma2), %g (eta_t) and %g (eta_s) are obtained." % (timeit.default_timer()-beginning, sigma2, eta_t, eta_s))
            beginning=timeit.default_timer()
            # stpo.prior.mean=u # re-center the prior
            print("Running the elliptic slice sampler (ESS) for %s prior model taking random seed %d ..." % ('STBP', args.seed_NO))
        
        if i==args.num_burnin:
            # start the timer
            tic=timeit.default_timer()
            print('\nBurn-in completed; recording samples now...\n')
        
        # update u
        if pre_optim:
            u=stpo.get_MAP(param0=u,PRINT=False,options=options)#, NCG=True)
            nl_u=stpo._get_misfit(u,MF_only=False)
        else:
            # generate MCMC sample with given sampler
            u,l_u=ESS(u,logLik(u),rnd_pri,logLik)
        u_=u-(0 if stpo.prior.mean is None else stpo.prior.mean)
        
        # update sigma2
        proj_u=stpo.prior.C_act(u_, -1.0/stpo.prior.bsv.q).reshape((stpo.prior.J,-1))
        quad=.5*np.sum(stpo.prior.qep.logpdf(proj_u,out='norms')**(2/stpo.prior.qep.q))
        pos_a, pos_b = a + stpo.prior.L*stpo.prior.J/2, b + quad
        if pre_optim:
            # optimize sigma2
            # sigmaq = pos_b/(pos_a+1)
            sigmaq = sigma2**(stpo.prior.qep.q/2)
            nl_sigma2 = -stats.invgamma.logpdf(sigmaq, a=pos_a, scale=pos_b)
        else:
            # sample sigma2
            # sigmaq = stats.invgamma.rvs(pos_a, scale=pos_b)
            sigmaq = sigma2**(stpo.prior.qep.q/2)
        sigma2 = sigmaq**(2/stpo.prior.qep.q)
        # update qep
        stpo.prior.qep = stpo.prior.qep.update(sigma2=sigma2)
        
        # update eta_t
        def logp_eta_t(eta_t, m=m_t, V=V_t):
            stpo.prior.qep = stpo.prior.qep.update(l=np.exp(eta_t))
            loglik = stpo.prior.logpdf(stpo.prior.vec2fun(u_))[0]
            # loglik = stpo.prior.qep.logpdf(stpo.prior.C_act(u_,-1.0/stpo.prior.bsv.q).swapaxes(0,1).reshape((stpo.prior.J,-1),order='F'))[0]
            # loglik = -stpo.prior.cost(u)
            logpri = -.5*(eta_t-m)**2/V
            return loglik+logpri
        if pre_optim:
            res = optimize.minimize(lambda x:-logp_eta_t(x), eta_t, method='L-BFGS-B', options=options)
            eta_t = res.x[0]; nl_eta_t = np.float64(res.fun)
        else:
            eta_t, l_eta_t = slice(eta_t,logp_eta_t(eta_t),logp_eta_t)
        # update qep
        stpo.prior.qep = stpo.prior.qep.update(l=np.exp(eta_t))
        
        # update eta_s
        def logp_eta_s(eta_s, m=m_s, V=V_s):
            stpo.prior.bsv = stpo.prior.bsv.update(l=np.exp(eta_s))
            loglik = stpo.prior.logpdf(stpo.prior.vec2fun(u_))[0]
            # loglik = stpo.prior.qep.logpdf(stpo.prior.C_act(u_,-1.0/stpo.prior.bsv.q).swapaxes(0,1).reshape((stpo.prior.J,-1),order='F'))[0]
            # loglik = -stpo.prior.cost(u)
            logpri = -.5*(eta_s-m)**2/V
            return loglik+logpri
        if pre_optim:
            res = optimize.minimize(lambda x:-logp_eta_s(x), eta_s, method='L-BFGS-B', options=options)
            eta_s = res.x[0]; nl_eta_s = np.float64(res.fun)
        else:
            eta_s, l_eta_s = slice(eta_s,logp_eta_s(eta_s),logp_eta_s)
        # update bsv
        stpo.prior.bsv = stpo.prior.bsv.update(l=np.exp(eta_s))
        
        # output some info
        if pre_optim:
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}'.format(i+1, nl_u, sigma2, nl_sigma2, eta_t, nl_eta_t, eta_s, nl_eta_s))
        else:
            # display acceptance at intervals
            if i+1 in prog:
                print('{0:.0f}% has been completed.'.format(np.float(i+1)/(args.num_samp+args.num_burnin)*100))
            
            # save results
            loglik.append([l_u,l_eta_t,l_eta_s])
            samp_sigma2[i] = sigma2
            samp_eta[i] = eta_t, eta_s
            if i>=args.num_burnin: samp_u.append(u)
            times.append(timeit.default_timer()-beginning)
    # stop timer
    toc=timeit.default_timer()
    time_=toc-tic
    print("\nAfter %g seconds, %d samples have been collected. \n" % (time_,args.num_samp))
    
    # store the results
    samp_u=np.stack(samp_u); loglik=np.stack(loglik);  times=np.stack(times)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'result')
    if not os.path.exists(savepath): os.makedirs(savepath)
    filename='stpo_ESS_hp_dim'+str(len(u))+'_'+ctime
    np.savez_compressed(os.path.join(savepath,filename), data_args=data_args, spat_args=spat_args, temp_args=temp_args, args=args, 
                        samp_u=samp_u, samp_eta=samp_eta, samp_sigma2=samp_sigma2, loglik=loglik,time_=time_,times=times)
    
    # plot
    # loaded=np.load(os.path.join(savepath,filename+'.npz'))
    # samp_u=loaded['samp_u']
    
    try:
        if stpo.prior.space=='vec': samp_u=stpo.prior.vec2fun(samp_u.T).T
        med_f = np.rot90(np.median(samp_u,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
        mean_f = np.rot90(np.mean(samp_u,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
        std_f = np.rot90(np.std(samp_u,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
    except Exception as e:
        print(e)
        mean_f=0; std_f=0
        n_samp=samp_u.shape[0]
        for i in range(n_samp):
            samp_i=stpo.prior.vec2fun(samp_u[i]) if stpo.prior.space=='vec' else samp_u[i]
            mean_f+=samp_i/n_samp
            std_f+=samp_i**2/n_samp
        std_f=np.sqrt(std_f-mean_f**2)
        mean_f=np.rot90(mean_f.reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
        std_f=np.rot90(std_f.reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
        med_f=None
    if med_f is not None:
        stpo.misfit.plot_reconstruction(rcstr_imgs=med_f, save_imgs=True, save_path='./reconstruction/ESS_hp_median')
    stpo.misfit.plot_reconstruction(rcstr_imgs=mean_f, save_imgs=True, save_path='./reconstruction/ESS_hp_mean')
    stpo.misfit.plot_reconstruction(rcstr_imgs=std_f, save_imgs=True, save_path='./reconstruction/ESS_hp_std')

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