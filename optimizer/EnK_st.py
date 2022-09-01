#!/usr/bin/env python
"""
(Regularizing) Ensemble Kalman Methods
Shiwei Lan @CalTech, 2017; @ASU, 2020
------------------------------------------------------------------
EKI: Algorithm 1 of 'Ensemble Kalman methods for inverse problems'
by Marco A Iglesias, Kody J H Law and Andrew M Stuart, Inverse Problems, Volume 29, Number 4, 2013
(Algorithm 1 of 'A regularizing iterative ensemble Kalman method for PDE-constrained inverse problems'
by Marco A Iglesias, Inverse Problems, Volume 32, Number 2, 2016)
EKS: Algorithm of 'Interacting Langevin Diffusions: Gradient Structure And Ensemble Kalman Sampler'
by Alfredo Garbuno-Inigo, Franca Hoffmann, Wuchen Li, and Andrew M. Stuart, SIAM Journal on Applied Dynamical Systems, Volume 19, Issue 1, February 4, 2020
-----------------------------
Created November 26, 2017
-----------------------------
Modified August 30, 2022 for spatiotemporal observations
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2017"
__license__ = "GPL"
__version__ = "0.6"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@gmail.com; slan@asu.edu"

import numpy as np
import scipy.sparse as sps
import timeit,time

class EnK(object):
    def __init__(self,u,G,misfit,prior=None,stp_sz=None,nz_lvl=1,alg='EKI',**kwargs):
        '''
        Ensemble Kalman Methods (EKI & EKS)
        -----------------------------------
        u: J ensemble states in D dimensional space
        G: forward mapping
        misfit: misfit object
        prior: prior object
        stp_sz: step size for discretized Kalman dynamics
        nz_lvl: level of perturbed noise to data, 0 means no perturbation
        reg: indicator of whether to implement regularization in optimization (EKI)
        adpt: indicator of whether to implement adaptation in stepsize (EKS)
        alg: algorithm option for choice of 'EKI' or 'EKS'
        optional:
        err_thld: threshold factor for stopping criterion
        reg_thld: threshold factor for finding regularizing parameter
        adpt_par: parameter in time-step adaptation to avoid overfloating
        '''
        # ensemble states
        self.u=u
        self.J,self.D=self.u.shape # number of ensembles (J) and state dimension (D)
        # forward mapping
        self.G=G
        # misfit and prior
        self.misfit=misfit
        self.prior=prior
        
        # algorithm specific parameters
        self.h=stp_sz
        self.r=nz_lvl
        self.alg=alg
        self.tau=kwargs.pop('err_thld',.1) # default to be .1
        self.reg=kwargs.pop('reg',False) # default to be false
        if self.reg and self.alg=='EKI':
            self.rho=kwargs.pop('reg_thld',0.5) # default to be 0.5
            self.tau=max(self.tau,1/self.rho) # tau >=1/rho
        self.adpt=kwargs.pop('adpt',True) # default to be true
        if self.adpt:
            self.eps=kwargs.pop('adpt_par',self.tau)
    
    # update ensemble sates
    def update(self):
        '''
        One step update of ensemble Kalman methods
        '''
        # prediction step
        p=self.G(self.u) # (J,m,T) where (m,T) is the data (observation) dimension
        p_m=np.mean(p,axis=0)
        n_T=p.shape[-1]
        
        # discrepancy principle
        eta=self.misfit.noise()*np.sqrt(self.r) if self.alg=='EKI' and self.r>0 else 0
        y_eta=self.misfit.data+eta; # perturb data by noise eta
        err=np.sqrt(self.misfit.cost(obs=p_m-eta))
        
        # analysis step
        p_tld=p-p_m
        p_tld=np.reshape(p_tld,(self.J,-1), order='F')
        # C_pp=[p_tld[:,:,t].T.dot(p_tld[:,:,t])/(self.J-1) for t in range(n_T)] # each (m,m)
        # C_pp_act=lambda v:np.stack([p_tld[:,:,t].T.dot(p_tld[:,:,t].dot(v[t]))/(self.J-1) for t in range(n_T)]) # (T,m,?)
        C_pp_act=lambda v: p_tld.T.dot(p_tld.dot(v))/(self.J-1)
        # C_up=[self.u.T.dot(p_tld[:,:,t])/(self.J-1) for t in range(n_T)] # each (D,m)
        u_=self.u.copy()
        # C_up_act=lambda v:np.stack([u_.T.dot(p_tld[:,:,t].dot(v[t]))/(self.J-1) for t in range(n_T)]) # (T,D,?)
        C_up_act=lambda v: u_.T.dot(p_tld.dot(v))/(self.J-1)
        # p_tld=None
        # Sherman–Morrison-Woodbury formula: inv(A+UCU.T) = invA - invA*U*inv(invC+U.T*invA*U)*U.T*invA
        SMW_act=lambda A,invC,U,solver: lambda v: solver(A,v) - solver(A, U.dot(np.linalg.solve(invC+U.T.dot(solver(A,U)), U.T.dot(solver(A,v)))))
        
        alpha={'EKI':1./self.h,'EKS':self.h}[self.alg]
        solver=sps.linalg.spsolve if sps.issparse(self.misfit.nzcov) else np.linalg.solve
        while self.reg and self.alg=='EKI':
            alpha*=2
            # t0=time.time()
            # err_alpha0=np.stack([np.linalg.solve(C_pp[t]+alpha*self.misfit.nzcov,(y_eta-p_m)[:,t]) for t in range(n_T)]).T
            # t1=time.time()
            # print('Time used is %.4f' % (t1-t0))
            # t0=time.time()
            # err_alpha=np.stack([SMW_act(alpha*self.misfit.nzcov, sps.eye(self.J), p_tld[:,:,t].T/np.sqrt(self.J), solver)((y_eta-p_m)[:,t]) for t in range(n_T)]).T
            err_alpha=SMW_act(alpha*sps.block_diag((self.misfit.nzcov,)*int(self.misfit.sz_t)), sps.eye(self.J), p_tld.T/np.sqrt(self.J), solver)(y_eta-p_m).T
            # t1=time.time()
            # print('Time used is %.4f' % (t1-t0))
            # print('Difference: %.6f' % (abs(err_alpha-err_alpha0).max()))
            if alpha*np.sqrt(self.misfit.cost(obs=self.misfit.data+err_alpha))>=self.rho*err: break
            err_alpha=None
        
        # t0=time.time()
        # d0=[np.linalg.solve(C_pp[t]+alpha*self.misfit.nzcov,(y_eta-p)[:,:,t].T) for t in range(n_T)] # each (m,J)
        # t1=time.time()
        # print('Time used is %.4f' % (t1-t0))
        # t0=time.time()
        # d=np.stack([SMW_act(alpha*self.misfit.nzcov, sps.eye(self.J), p_tld[:,:,t].T/np.sqrt(self.J), solver)((y_eta-p)[:,:,t].T) for t in range(n_T)]) # each (m,J)
        d=SMW_act(alpha*sps.block_diag((self.misfit.nzcov,)*int(self.misfit.sz_t)), sps.eye(self.J), p_tld.T/np.sqrt(self.J), solver)((y_eta-p).reshape((self.J,-1), order='F').T)
        # t1=time.time()
        # print('Time used is %.4f' % (t1-t0))
        # print('Difference: %.6f' % (abs(np.stack(d)-np.stack(d0)).max()))
        
        if self.alg=='EKI':
            # C_pp=None
            # self.u+=np.sum([C_up[t].dot(d[t]) for t in range(n_T)],axis=0).T
            # self.u+=np.sum(C_up_act(d),axis=0).T
            self.u+=C_up_act(d).T
        elif self.alg=='EKS':
            # if self.adpt: alpha/=np.sqrt(np.sum([d[t]*C_pp[t].dot(d[t]) for t in range(n_T)])*(self.J-1))*alpha+self.eps
            if self.adpt: alpha/=np.sqrt(np.sum(d*C_pp_act(d))*(self.J-1))*alpha+self.eps
            # C_pp=None
            u_tld=self.u-np.mean(self.u,axis=0)
            if self.D<=2*self.J:
                # C_uu=np.cov(self.u,rowvar=False)
                # self.u=self.prior.cov.dot(np.linalg.solve(self.prior.cov+alpha*C_uu,self.u.T+np.sum([C_up[t].dot(d[t]) for t in range(n_T)],axis=0))).T
                # self.u=self.prior.cov.dot(np.linalg.solve(self.prior.cov+alpha*np.cov(self.u,rowvar=False),self.u.T+np.sum([C_up[t].dot(d[t]) for t in range(n_T)],axis=0))).T
                # self.u=self.prior.cov.dot(np.linalg.solve(self.prior.cov+alpha*np.cov(self.u,rowvar=False),self.u.T+np.sum(C_up_act(d),axis=0))).T
                self.u=self.prior.cov.dot(np.linalg.solve(self.prior.cov+alpha*np.cov(self.u,rowvar=False),self.u.T+C_up_act(d))).T
            else:
                # use Sherman–Morrison-Woodbury formula for D>>J
                # self.u=self.u.T+np.sum([C_up[t].dot(d[t]) for t in range(n_T)],axis=0)
                # self.u=self.u.T+np.sum(C_up_act(d),axis=0)
                self.u=self.u.T+C_up_act(d)
                solver=sps.linalg.spsolve if sps.issparse(self.prior.cov) else np.linalg.solve
                # self.u-=u_tld.T.dot(np.linalg.solve(self.J/alpha*np.eye(self.J)+u_tld.dot(solver(self.prior.cov,u_tld.T)), u_tld.dot(solver(self.prior.cov,self.u))))
                self.u=self.prior.cov.dot(SMW_act(self.prior.cov, sps.eye(self.J)/alpha, u_tld.T/np.sqrt(self.J), solver)(self.u))
                self.u=self.u.T
            self.u+=np.random.randn(self.J,self.J).dot(u_tld)*np.sqrt(2*alpha/self.J)
        
        return err,p
    
    # run EnK
    def run(self,max_iter=100,SAVE=False):
        '''
        Run ensemble Kalman methods to collect ensembles estimates/samples
        '''
        print('\nRunning '+self.alg+' now...\n')
        if self.h is None: self.h=1./max_iter
        errs=np.zeros(max_iter)
        fwdouts=[]
        ensbls=[]
        ensbls.append(self.u) # record the initial ensemble
        u_est=np.zeros((max_iter,self.D))
        # start the timer
        tic=timeit.default_timer()
        r=self.r if self.r>0 else 1
        for n in range(max_iter):
            # update the Kalman filter
            errs[n],fwdout_n=self.update()
            if SAVE=='all':
                fwdouts.append(fwdout_n)
                ensbls.append(self.u) # record the ensemble
            # estimate unknown parameters
            u_est[n]=np.mean(self.u,axis=0)
            print('Estimated unknown parameters: '+(min(self.D,10)*"%.4f ") % tuple(u_est[n,:min(self.D,10)]) )
            print(self.alg+' at iteration %d, with error %.8f.\n' % (n+1,errs[n]) )
            # terminate if discrepancy principle satisfied
            if errs[n]<=self.tau*r: break
        # stop timer
        toc=timeit.default_timer()
        t_used=toc-tic
        print('EnK terminates at iteration %d, with error %.4f, using time %.4f.' % (n+1,errs[n],t_used) )
        
        return_list=u_est,errs,n,t_used
        if SAVE:
            if SAVE=='all': return_list+=(np.stack(fwdouts),np.stack(ensbls))
            return self.save(return_list)
        else:
            return return_list
    
    # save results to file
    def save(self,dump_list):
        import os,errno
        import pickle
        # create folder
        cwd=os.getcwd()
        savepath=os.path.join(cwd,'result')
        try:
            os.makedirs(savepath)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
            else:
                raise
        # name file
        ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
        filename=self.alg+'_ensbl'+str(self.J)+'_dim'+str(self.D)+'_'+ctime
        # dump data
        f=open(os.path.join(savepath,filename+'.pckl'),'wb')
        pickle.dump(dump_list,f)
        f.close()
        
        return savepath,filename
    
#         # load data
#         f=open(os.path.join(savepath,filename+'.pckl'),'rb')
#         loaded=pickle.load(f)
#         f.close()
    
# test
# if __name__=='__main__':