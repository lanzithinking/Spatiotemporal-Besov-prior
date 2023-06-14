#!/usr/bin/env python
"""
Class definition of spatiotemporal Besov prior for the dynamic linear model.
--------------------------------------------------------------------------
Created June 30, 2022 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "1.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu lanzithinking@outlook.com"

import os,sys
import numpy as np
import scipy as sp
import scipy.sparse as sps

# self defined modules
sys.path.append( "../" )
from util.stbp.BSV import BSV
from util.stbp.qEP import qEP
from util.stbp.STBP import STBP

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class prior(STBP):
    """
    Spatiotemporal Besov prior measure STBP(mu,C,q).
    """
    def __init__(self,sz_x,sz_t,mean=None,store_eig=False,**kwargs):
        '''
        if not hasattr(sz_x, "__len__"):
            self.sz_x=(sz_x,)
        else:
            self.sz_x=sz_x # I = np.prod(sz_x)
        xx,yy=np.meshgrid(np.linspace(0,1,self.sz_x[0]),np.linspace(0,1,self.sz_x[1]))
        bsv=BSV(x=np.stack([xx.flatten(),yy.flatten()]).T,store_eig=store_eig,**kwargs.pop('spat_args',{}))
        '''
        self.sz_x, self.sz_t=sz_x, sz_t # J = sz_t
        x=self._grid(gdsz=self.sz_x)
        bsv=BSV(x=x,store_eig=store_eig,**kwargs.pop('spat_args',{}))
        
        ##!!!!!!!!!!!!!!!!!!!!!!?? does this need to be consistent with defination in misfit????????????????
        t=np.linspace(0,2,self.sz_t) 
        qep=qEP(x=t,store_eig=store_eig,**kwargs.pop('temp_args',{}))
        self.space=kwargs.pop('space','fun') # alternative 'fun'
        super().__init__(spat=bsv, temp=qep, store_eig=store_eig, **kwargs) # N = I*J
        self.dim={'vec':self.L*self.J,'fun':self.N}[self.space]
        self.mean=mean
        if self.mean is not None:
            assert self.mean.size==self.dim, "Non-conforming size of mean!"
    
    def _grid(self,gdsz=None):
        """
        Build the time grid
        """
        self.gdsz=gdsz
        # set the grid
        grid=np.linspace(0,1,self.gdsz)
        # print('\nThe grid is defined.')
        return grid
    
    def cost(self,u,logr=False):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        sum_l [-.5*N(q/2-1) log(r_l) + .5 r_l^(q/2)], r_l(u)^(q/2) = ||_C^(-1/q) u_l(x)||^q = |gamma_l^{-1} <phi_l, u>|^q
        """
        if u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None:
            u=u-self.mean
        
        proj_u=self.C_act(u, -1.0/self.bsv.q) # (LJ,)
        proj_u=proj_u.reshape((self.J,-1))
        norms=self.qep.logpdf(proj_u,out='norms')**(self.bsv.q/self.qep.q) # r^(q/2)
        val=-np.sum(np.log(norms))*self.N/2*(1-2/self.bsv.q)*logr + 0.5*np.sum(norms)
        return val
    
    # def grad(self,u):
    #     """
    #     Calculate the gradient of log-prior
    #     """
    #     if u.shape[0]!=self.dim:
    #         u=u.reshape((self.dim,-1),order='F')
    #     if self.mean is not None:
    #         u=u-self.mean
    #
    #     eigv, eigf=self.eigs()
    #     eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
    #     gamma=eigv**(1/self.bsv.q)
    #     proj_u=((u if self.space=='vec' else eigf.T.dot(u) if self.space=='fun' else ValueError('Wrong space!'))/gamma).reshape((self.J,-1)) # (J,L)
    #     qep_norm=self.qep.logpdf(proj_u,out='norms')**(1/self.qep.q) # (L,)
    #     g=0.5*self.bsv.q*(qep_norm**(self.bsv.q-2) *self.qep.solve(proj_u)).reshape((self.L*self.J,-1))/gamma[:,None] # (LJ,)
    #     if self.space=='fun': g=eigf.dot(g) # (N,)
    #     return g.squeeze()
    
    def grad(self,u,logr=False):
        """
        Calculate the gradient of log-prior
        """
        if u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None:
            u=u-self.mean
    
        # eigv, eigf=self.bsv.eigs()
        # eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
        # gamma=eigv**(1/self.bsv.q)
        # proj_u=((u.reshape((self.L,self.J,-1),order='F') if self.space=='vec' else eigf.T.dot(u.reshape((self.J,self.I,-1))) if self.space=='fun' else ValueError('Wrong space!'))/gamma[:,None,None]).swapaxes(0,1).reshape((self.J,-1),order='F') # (J,L)
        
        proj_u=self.C_act(u, -1.0/self.bsv.q).reshape((self.J,-1)) # (J,L_)
        qep_norm=self.qep.logpdf(proj_u,out='norms')**(1/self.qep.q) # r^(1/2), (L,)
        A=0.5*(-self.N*(self.bsv.q-2)*logr + self.bsv.q*qep_norm**self.bsv.q)/qep_norm**2
        g=(A *self.qep.solve(proj_u)/self.gamma).swapaxes(0,1) # (L,J)
        # if self.space=='fun': g=eigf.dot(g) # (I,J)
        g=g.reshape((self.dim,-1),order='F') if self.space=='vec' else self.vec2fun(g) if self.space=='fun' else ValueError('Wrong space!')
        return g.squeeze()
    
    def Hess(self,u,logr=False):
        """
        Calculate the Hessian action of log-prior
        """
        if u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None:
            u=u-self.mean
        
        proj_u=self.C_act(u, -1.0/self.bsv.q).reshape((self.J,-1)) # (J,L_)
        qep_norm=self.qep.logpdf(proj_u,out='norms')**(1/self.qep.q) # (L,)
        
        def hess(v):
            if v.shape[0]!=self.L*self.J: v=self.fun2vec(v)
            v=v.reshape((self.L,self.J,-1),order='F').swapaxes(0,1) # (J,L,K)
            A=.5*(-self.N*(self.bsv.q-2)*logr + self.bsv.q*qep_norm[None,:,None]**self.bsv.q)/qep_norm[None,:,None]**2
            B=(self.bsv.q-2)*(self.N*logr + self.bsv.q/2*qep_norm[None,:,None]**self.bsv.q)/qep_norm[None,:,None]**4
            Hv=A*self.qep.solve(v/self.gamma[None,:,None])/self.gamma[None,:,None]
            Hv+=B*self.qep.solve(proj_u)[:,:,None]*np.sum(proj_u[:,:,None]*self.qep.solve(v/self.gamma[None,:,None]),axis=0,keepdims=True)/self.gamma[None,:,None]
            Hv=Hv.swapaxes(0,1) # (L,J,K)
            Hv=Hv.reshape((self.dim,-1),order='F') if self.space=='vec' else self.vec2fun(Hv) if self.space=='fun' else ValueError('Wrong space!')
            return Hv.squeeze()
        return hess
    
    def invHess(self,u,logr=False):
        """
        Calculate the inverse Hessian action of log-prior
        """
        if u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None:
            u=u-self.mean
        
        proj_u=self.C_act(u, -1.0/self.bsv.q).reshape((self.J,-1)) # (J,L_)
        qep_norm=self.qep.logpdf(proj_u,out='norms')**(1/self.qep.q) # (L,)
        
        def ihess(v): # does not exist if q=1 (taking (q/2|xi|**(q-2))**(-1)C^(-1) )
            if v.shape[0]!=self.L*self.J: v=self.fun2vec(v)
            v=v.reshape((self.L,self.J,-1),order='F').swapaxes(0,1) # (J,L,K)
            A=.5*(-self.N*(self.bsv.q-2)*logr + self.bsv.q*qep_norm[None,:,None]**self.bsv.q)/qep_norm[None,:,None]**2
            B=(self.bsv.q-2)*(self.N*logr + self.bsv.q/2*qep_norm[None,:,None]**self.bsv.q)/qep_norm[None,:,None]**4
            C=0.5*(self.N*(self.bsv.q-2)*logr + self.bsv.q*(self.bsv.q-1)*qep_norm[None,:,None]**self.bsv.q)/qep_norm[None,:,None]**2
            iHv=self.qep.mult(v*self.gamma[None,:,None])*self.gamma[None,:,None]
            if np.any(C): iHv+=-B/C*proj_u[:,:,None]*np.sum(proj_u[:,:,None]*(v*self.gamma[None,:,None]),axis=0,keepdims=True)*self.gamma[None,:,None]
            iHv/=A
            iHv=iHv.swapaxes(0,1) # (L,J,K)
            iHv=iHv.reshape((self.dim,-1),order='F') if self.space=='vec' else self.vec2fun(iHv) if self.space=='fun' else ValueError('Wrong space!')
            return iHv.squeeze()
        return ihess
    
    def sample(self, output_space=None, mean=None):
        """
        Sample a random function u ~ STBP(0,_C)
        vector u ~ STBP(0,C,q): u = gamma xi phi, xi ~ qEP(0,C)
        """
        if output_space is None:
            output_space=self.space
        if mean is None:
            mean=self.mean
        
        if output_space=='vec':
            qep_rv=self.qep.rnd(n=self.L)*self.gamma[None,:] # (J,L)
            u=qep_rv.flatten() # (LJ,)
        elif output_space=='fun':
            u=super().rnd(n=1).squeeze() # (N,)
        else:
            raise ValueError('Wrong space!')
        if mean is not None:
            u+=mean
        return u
    
    # def C_act(self,u,comp=1):
    #     """
    #     Calculate operation of C^comp on vector u: u --> C^comp * u
    #     """
    #     if u.shape[0]!=self.dim:
    #         u=u.reshape((self.dim,-1),order='F')
    #
    #     if comp==0:
    #         return u
    #     else:
    #         eigv, eigf=self.eigs()
    #         if comp<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
    #         Cu=(u if self.space=='vec' else eigf.T.dot(u) if self.space=='fun' else ValueError('Wrong space!'))*eigv**(comp)
    #         return Cu
    
    def C_act(self,u,comp=1):
        """
        Calculate operation of C^comp on vector u: u --> C^comp * u
        """
        u_sz0={'vec':self.L,'fun':self.I}[self.space]
        if u.shape[0]!=u_sz0: u=u.reshape((u_sz0,self.J,-1),order='F')
        if np.ndim(u)==2: u=u[:,:,None]
    
        if comp==0:
            return u
        else:
            eigv, eigf=self.bsv.eigs()
            if comp<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
            # Cu=(u if self.space=='vec' else eigf.T.dot(u.swapaxes(0,1)) if self.space=='fun' else ValueError('Wrong space!'))*eigv[:,None,None]**(comp)
            Cu=(u if self.space=='vec' else np.tensordot(eigf,u,axes=(0,0)) if self.space=='fun' else ValueError('Wrong space!'))*eigv[:,None,None]**(comp)
            return Cu.reshape((self.L*self.J,-1),order='F')
    
    # def vec2fun(self, u_vec):
    #     """
    #     Convert vector (u_i) to function (u)
    #     """
    #     if u_vec.shape[0]!=self.L*self.J: u_vec=u_vec.reshape((self.L*self.J,-1),order='F')
    #     _, eigf = self.eigs()
    #     u_f = eigf.dot(u_vec)
    #     return np.squeeze(u_f)
    
    def vec2fun(self, u_vec):
        """
        Convert vector (u_i) to function (u)
        """
        if u_vec.shape[0]!=self.L: u_vec=u_vec.reshape((self.L,self.J,-1),order='F')
        if np.ndim(u_vec)==2: u_vec=u_vec[:,:,None]
        _, eigf = self.bsv.eigs()
        # u_f = eigf.dot(u_vec.swapaxes(0,1)).reshape((self.N,-1),order='F')
        u_f = np.tensordot(eigf,u_vec,axes=1).reshape((self.N,-1),order='F')
        return np.squeeze(u_f)
    
    # def fun2vec(self, u_f):
    #     """
    #     Convert vector (u_i) to function (u)
    #     """
    #     if u_f.shape[0]!=self.N: u_f=u_f.reshape((self.N,-1),order='F')
    #     _, eigf = self.eigs()
    #     u_vec = eigf.T.dot(u_f)
    #     return np.squeeze(u_vec)
    
    def fun2vec(self, u_f):
        """
        Convert vector (u_i) to function (u)
        """
        if u_f.shape[0]!=self.I: u_f=u_f.reshape((self.I,self.J,-1),order='F')
        if np.ndim(u_f)==2: u_f=u_f[:,:,None]
        _, eigf = self.bsv.eigs()
        # u_vec = eigf.T.dot(u_f.swapaxes(0,1)).reshape((self.L*self.J,-1),order='F')
        u_vec = np.tensordot(eigf,u_f,axes=(0,0)).reshape((self.L*self.J,-1),order='F')
        return np.squeeze(u_vec)
    
if __name__ == '__main__':
    np.random.seed(2022)
    # define the prior
    sz_x=2**4; sz_t=20
    spat_args={'basis_opt':'Fourier','sigma2':1,'l':1,'s':1.5,'q':1.01,'L':1000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    prior = prior(sz_x=sz_x, sz_t=sz_t, spat_args=spat_args, temp_args=temp_args, space='fun')
    # prior.mean = prior.sample()
    # generate sample
    u=prior.sample()
    logr=False
    # u=np.random.rand(prior.N,)
    nlogpri=prior.cost(u,logr=logr)
    ngradpri=prior.grad(u,logr=logr)
    print('The negative logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(nlogpri,np.linalg.norm(ngradpri)))
    hess=prior.Hess(u,logr=logr)
    # test
    h=1e-7
    v=prior.sample()
    ngradv_fd=(prior.cost(u+h*v,logr=logr)-nlogpri)/h
    ngradv=ngradpri.dot(v)
    rdiff_gradv=np.abs(ngradv_fd-ngradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    hessv_fd=(prior.grad(u+h*v,logr=logr)-ngradpri)/h
    hessv=hess(v)
    rdiff_hessv=np.linalg.norm(hessv_fd-hessv)/np.linalg.norm(v)
    print('Relative difference of Hessian-action in a random direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
    ihess=prior.invHess(u,logr=logr)
    v1=ihess(hessv)
    rdiff_v1=np.linalg.norm(v1-v)/np.linalg.norm(v)
    print('Relative difference of invHessian-Hessian-action in a random direction between the composition and identity: %.10f' % rdiff_v1)
    v2=hess(ihess(v))
    rdiff_v2=np.linalg.norm(v2-v)/np.linalg.norm(v)
    print('Relative difference of Hessian-invHessian-action in a random direction between the composition and identity: %.10f' % rdiff_v2)
    # plot
    import matplotlib.pyplot as plt
    if u.shape[0]!=prior.N: u=prior.vec2fun(u)
    u=u.reshape((-1,prior.sz_t),order='F')
    fig, axes=plt.subplots(nrows=1,ncols=4,sharex=True,sharey=True,figsize=(15,4))
    n_j=int(np.floor(sz_t/4))
    for j,t_j in enumerate(prior.qep.x[::n_j,0]):
        ax=axes.flat[j]
        ax.imshow(u[:,j*n_j].reshape(prior.sz_x),origin='lower',extent=[0,1,0,1])
        ax.set_title('t = %.2f'% (t_j,))
    plt.show()
    