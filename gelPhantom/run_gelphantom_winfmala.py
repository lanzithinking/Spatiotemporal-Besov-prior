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
from gelPhantom import gelPhantom

# MCMC
import sys
sys.path.append( "../" )
from sampler.winfMALA import winfMALA


np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=2000)    
    parser.add_argument('alg_NO', nargs='?', type=int, default=0)
    parser.add_argument('step_sizes', nargs='?', type=float, default=(9e-8,))
    parser.add_argument('algs', nargs='?', type=str, default=('winfMALA',))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    np.random.seed(seed)
    # define Bayesian inverse problem
    spat_args={'basis_opt':'Fourier','l':1,'s':1,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    store_eig = True
    gph = gelPhantom(spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed, init_param=True)
    # logLik = lambda u: -gph._get_misfit(u, MF_only=True)
    # transformation
    nmlz = lambda z,q=1: z/np.linalg.norm(z,axis=1)[:,None]**q
    # Lmd = lambda z,q=gph.prior.qep.q: gph.prior.qep.act(nmlz(z.reshape((-1,gph.prior.qep.N),order='F'),1-2/q),alpha=0.5,transp=True)
    def Lmd(z, q=gph.prior.qep.q, grad=False):
        _z = z.reshape((-1,gph.prior.qep.N),order='F') # (L,J)
        nm_z = np.linalg.norm(_z,axis=1)[:,None]
        lmd = gph.prior.qep.act(_z*nm_z**(2/q-1),alpha=0.5,transp=True)#,chol=False) # (L,J)
        if grad:
            # return lambda v: gph.prior.qep.act(_z*z.dot(v)*nm_z**(2/q-3)*(2/q-1) + v.reshape((-1,gph.prior.qep.N),order='F')*nm_z**(2/q-1), alpha=0.5,transp=True)
            def dlmd(v, adj=False):
                _v = v.reshape((-1,gph.prior.qep.N),order='F')
                if adj:
                    _v = gph.prior.qep.act(_v, alpha=0.5,transp=True,adjt=adj)#,chol=False)
                    return _z*np.sum(_z*_v,axis=1)[:,None]*nm_z**(2/q-3)*(2/q-1) + _v*nm_z**(2/q-1)
                else:
                    return gph.prior.qep.act(_z*np.sum(_z*_v,axis=1)[:,None]*nm_z**(2/q-3)*(2/q-1) + _v*nm_z**(2/q-1), alpha=0.5,transp=True)#,chol=False)
            return dlmd
        else:
            return lmd
    # T = lambda z,q=gph.prior.bsv.q: gph.prior.C_act(Lmd(z), 1/q)
    def T(z, q=gph.prior.bsv.q, grad=False):
        if grad:
            return lambda v,adj=False: Lmd(z, grad=grad)(gph.prior.C_act(v, 1/q),adj=adj).flatten(order='F') if adj else gph.prior.C_act(Lmd(z, grad=grad)(v), 1/q)
        else:
            return gph.prior.C_act(Lmd(z), 1/q)
    invLmd = lambda xi,q=gph.prior.qep.q: nmlz(gph.prior.qep.act(xi.reshape((-1,gph.prior.qep.N),order='F'),alpha=-0.5,transp=True),1-q/2)
    invT = lambda u,q=gph.prior.bsv.q: invLmd(gph.prior.C_act(u, -1/q))
    # log-likelihood
    def logLik(u, T=None, grad=False):
        parameter = T(u) if callable(T) else u
        geom_ord=[0]+[int(grad)]
        l,g = gph.get_geom(parameter, geom_ord, MF_only=True)[:2]
        if grad:
            if callable(T): g = T(u, grad=True)(g,adj=True)
            return l,g.squeeze()
        else:
            return l
    
    # initialization random noise epsilon
    try:
        # fld=os.path.join(os.getcwd(),'reconstruction/MAP')
        # with open(os.path.join(fld,'MAP.pckl'),'rb') as f:
        #     map=pickle.load(f)
        # f.close()
        # u=invT(map).flatten(order='F')
        u_init=gph.init_parameter if hasattr(gph,'init_parameter') else gph._init_param()
        u=invT(u_init).flatten(order='F')
    except Exception as e:
        print(e)
        u=np.random.randn({'vec':gph.prior.L*gph.prior.qep.N,'fun':gph.prior.N}[gph.prior.space])
    l,g=logLik(u,T,grad=True)
    # # test dlogLik
    # h=1e-8; v=gph.prior.sample()
    # l1=logLik(u+h*v,T)
    # print('test error: %0.4f' %(abs((l1-l)/h-g.dot(v))/np.linalg.norm(v)))
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for random seed %d..."
          % (args.algs[args.alg_NO],args.step_sizes[args.alg_NO],args.seed_NO))
    
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
        
        u,l,g,acpt_ind=winfMALA(u,l,g,logLik,T,args.step_sizes[args.alg_NO])
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
    filename='gph_'+args.algs[args.alg_NO]+'_dim'+str(len(u))+'_'+ctime
    np.savez_compressed(os.path.join(savepath,filename),spat_args=spat_args, temp_args=temp_args, args=args, samp=samp,loglik=loglik,time_=time_,times=times)
    
    # plot
    # loaded=np.load(os.path.join(savepath,filename+'.npz'))
    # samp=loaded['samp']
    
    mcmc_v_med = np.median(samp,axis=0)
    mcmc_v_mean = np.mean(samp,axis=0)
    mcmc_f = gph.prior.vec2fun(mcmc_v_med).reshape(np.append(gph.misfit.sz_x,gph.misfit.sz_t),order='F').swapaxes(0,1)
    gph.misfit.plot_reconstruction(rcstr_imgs=mcmc_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_median')
    mcmc_f = gph.prior.vec2fun(mcmc_v_mean).reshape(np.append(gph.misfit.sz_x,gph.misfit.sz_t),order='F').swapaxes(0,1)
    gph.misfit.plot_reconstruction(rcstr_imgs=mcmc_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.alg_NO]+'_mean')


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