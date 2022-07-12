#!/usr/bin/env python
"""
spatiotemporal Besov process
-- with basis choices 'wavelet' and 'Fourier'
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ ASU, 2022
-------------------------------
Created June 13, 2022 @ ASU
-------------------------------
https://github.com/lanzithinking/Spatiotemporal-Besov-prior
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, STBP project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.stats import gennorm
# self defined modules
import sys
sys.path.append( "../../" )
from util.stbp.BSV import *
from util.stbp.EPP import *
from util.stbp.linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class STBP(BSV):
    def __init__(self,spat,temp,store_eig=False,**kwargs):
        """
        Initialize the STBP class with spatial (BSV) class bsv, temporal (EPP) class epp and the dynamic eigenvalues Lambda
        spat: spatial class (discrete size I x L)
        temp: temporal class (discrete size J x L)
        gamma: decaying eigenvalues
        store_eig: indicator to store eigen-pairs, default to be false
        spdapx: use speed-up or approximation
        -----------------------------------------------
        u(x,t) = sum_{l=1}^infty lambda_l(t) * phi_l(x)
        lambda_l = gamma_l xi_l(t) ~ EPP(0,cov_l,q), xi_l ~ EPP(0, C, q)
        phi_l - basis of Besov prior
        """
        if type(spat) is BSV:
            self.bsv=spat # spatial class
        else:
            self.bsv=BSV(spat,store_eig=store_eig,**kwargs)
        if type(temp) is EPP:
            self.epp=temp # temporal class
        else:
            self.epp=EPP(temp,store_eig=store_eig or Lambda is None,**kwargs)
        self.parameters=kwargs # all parameters of the kernel
        self.I,self.J=self.bsv.N,self.epp.N # spatial and temporal dimensions
        self.N=self.I*self.J # joint dimension (number of total inputs per trial)
        self.L=self.bsv.L#*self.epp.L
        if self.L>self.I:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed the size of spatial basis!")
            self.L=self.I
        self.gamma=pow(np.arange(1,self.L+1),-(self.bsv.s/self.bsv.d+1./2-1./self.bsv.q))
        try:
            self.comm=MPI.COMM_WORLD
        except:
            print('Parallel environment not found. It may run slowly in serial.')
            self.comm=None
        self.spdapx=self.parameters.get('spdapx',self.N>1e3)
        self.store_eig=store_eig
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self.eigs(**kwargs)
    
    def tomat(self,**kwargs):
        """
        Get kernel as matrix
        """
        alpha=kwargs.get('alpha',1)
        eigv_=pow(self.gamma**self.bsv.q,alpha)
        if alpha<0: eigv_[abs(eigv_)<np.finfo(float).eps]=np.finfo(float).eps
        _,Phi_x=self.bsv.eigs(self.L)
        C_x=(Phi_x*eigv_[None,:]).dot(Phi_x.T) + ((alpha>=0)*self.bsv.jit)*sps.eye(self.I)
        C=sps.block_diag((sps.csr_matrix(C_x),)*self.J,format='csr') # (IJ,IJ)
        # if self.spdapx and not sps.issparse(C):
        #     warnings.warn('Possible memory overflow!')
        return C
    
    # def tomat(self,**kwargs):
    #     """
    #     Get kernel as matrix
    #     """
    #     alpha=kwargs.get('alpha',1)
    #     C = np.kron(self.epp.tomat(alpha=alpha),self.bsv.tomat(alpha=alpha)) # (IJ,IJ)
    #     # if self.spdapx and not sps.issparse(C):
    #     #     warnings.warn('Possible memory overflow!')
    #     return C
    
    def mult(self,v,**kwargs):
        """
        Kernel multiply a function (vector): C*v
        """
        alpha=kwargs.pop('alpha',1) # power of dynamic eigenvalues
        if not self.spdapx:
            if v.shape[0]!=self.N:
                v=v.reshape((self.N,-1),order='F')
            Cv=self.tomat(alpha=alpha).dot(v)
        else:
            eigv,eigf = self.eigs() # obtain eigen-pairs
            if alpha<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
            eigv_ = pow(eigv,alpha)
            prun=kwargs.get('prun',True) and self.comm # control of parallel run
            if prun:
                try:
#                     import pydevd; pydevd.settrace()
                    nproc=self.comm.size; rank=self.comm.rank
                    if nproc==1: raise Exception('Only one process found!')
                    Cv_loc=multf(eigf[rank::nproc,:].multiply(eigv_),multf(eigf.T,v))
                    Cv=np.empty_like(v)
                    self.comm.Allgatherv([Cv_loc,MPI.DOUBLE],[Cv,MPI.DOUBLE])
                    pidx=np.concatenate([np.arange(self.N)[i::nproc] for i in np.arange(nproc)])
                    Cv[pidx]=Cv.copy()
                except Exception as e:
                    if rank==0:
                        warnings.warn('Parallel run failed: '+str(e))
                    prun=False
                    pass
            if not prun:
#                 Cv=np.concatenate([multf(eigf_i*eigv,multf(eigf.T,v)) for eigf_i in eigf])
                Cv=multf(eigf.multiply(eigv_),multf(eigf.T,v))
            Cv+=self.bsv.jit*v
        return Cv
    
    # def mult(self,v,**kwargs):
    #     """
    #     Kernel multiply a function (vector): C*v
    #     """
    #     alpha=kwargs.pop('alpha',1) # power of dynamic eigenvalues
    #     if not self.spdapx:
    #         if v.shape[0]!=self.N:
    #             v=v.reshape((self.N,-1),order='F')
    #         Cv=self.tomat(alpha=alpha).dot(v)
    #     else:
    #         if v.shape[0]!=self.I:
    #             v=v.reshape((self.I,self.J,-1),order='F')
    #         if np.ndim(v)<3: v=v[:,:,None]
    #         Cv=self.epp.act(self.bsv.mult(v,alpha=alpha),alpha=alpha,transp=True).reshape((self.N,-1),order='F') # (IJ,K_)
    #     return Cv
    
    def solve(self,v,**kwargs):
        """
        Kernel solve a function (vector): C^(-1)*v
        """
        return BSV.solve(self,v,**kwargs)
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis
        C * eigf_i = eigf_i * eigv_i, i=1,...,L
        """
        if L is None:
            L=self.L;
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            L=min(L,self.N)
            gamma=self.gamma[:L] if L <=self.L else pow(np.arange(1,L+1),-(self.bsv.s/self.bsv.d+1./2-1./self.bsv.q))
            eigv=np.tile(gamma**self.bsv.q,self.J) # (LJ,)
            _,Phi_x=self.bsv.eigs(L);
            eigf=sps.kron(sps.eye(self.J),Phi_x).tocsc() # (IJ,LJ)
        else:
            eigv,eigf=self.eigv,self.eigf
            if L<self.L:
                eigv=eigv.reshape((-1,self.L))[:,:L].flatten(); # (LJ,1)
                eigf=eigf.reshape((self.N,-1,self.L))[:,:,:L].reshape((self.N,-1)) # (IJ,LJ)
        return eigv,eigf
    
    # def eigs(self,L=None,upd=False,**kwargs):
    #     """
    #     Obtain partial eigen-basis
    #     C * eigf_i = eigf_i * eigv_i, i=1,...,L
    #     """
    #     if L is None:
    #         L=self.L;
    #     if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
    #         L=min(L,self.N)
    #         lambda_t,Phi_t=self.epp.eigs(); lambda_x,Phi_x=self.bsv.eigs()
    #         eigv=np.kron(lambda_t,lambda_x); eigf=np.kron(Phi_t,Phi_x) # (IJ,L)
    #         if L<=self.bsv.L*self.epp.L:
    #             eigv=eigv[:L]; eigf=eigf[:,:L]
    #         else:
    #             warnings.warn('Requested too many eigenvalues!')
    #     else:
    #         eigv,eigf=self.eigv,self.eigf
    #         if L<self.L:
    #             eigv=eigv[:L]; eigf=eigf[:,:L]
    #     return eigv,eigf
    
    def act(self,x,alpha=1,**kwargs):
        """
        Obtain the action of C^alpha
        y=C^alpha *x
        """
        return BSV.act(self,x,alpha=alpha,**kwargs)
    
    def logdet(self):
        """
        Compute log-determinant of the kernel C: log|C|
        """
        return BSV.logdet(self)
    
    def logpdf(self,X):
        """
        Compute logpdf of centered spatiotemporal Besov process X ~ STBP(0,C,q)
        """
        if not self.spdapx:
            logpdf,q_ldet=BSV.logpdf(self,self.epp.act(X.reshape((self.J,self.I,-1)).reshape((self.N,-1)),alpha=-.5))
        else:
            q_ldet=-X.shape[1]*self.logdet()/self.bsv.q
            _,Phi_x=self.bsv.eigs();
            proj_X=Phi_x.T.dot(X.reshape((self.J,self.I,-1)))/self.gamma[:,None,None] # (L,J,K_)
            proj_X=proj_X.reshape((self.J,-1))
            epp_norm=self.epp.logpdf(proj_X,out='norms')
            qsum=-0.5*np.sum(epp_norm**(self.bsv.q/self.epp.q))
            logpdf=q_ldet+qsum
        return logpdf,q_ldet
    
    def update(self,bsv=None,epp=None):
        """
        Update the eigen-basis
        """
        if bsv is not None:
            self.bsv=bsv; self.I=self.bsv.N; self.N=self.I*self.J
        if epp is not None:
            self.epp=C_t; self.J=self.epp.N; self.N=self.I*self.J
        if self.store_eig:
            self.eigv,self.eigf=self.eigs(upd=True)
        return self
    
    def rnd(self,n=1):
        """
        Generate spatiotemporal Besov random function (vector) rv ~ STBP(0,C,q)
        """
        epp_rv=self.epp.rnd(n=self.L*n).reshape((-1,self.L,n),order='F')*self.gamma[None,:,None] # (J,L,n)
        _,Phi_x=self.bsv.eigs();
        rv=Phi_x.dot(epp_rv).reshape((-1,n),order='F') # (N,n)
        return rv

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    # L=20
    ## spatial class
    # x=np.random.rand(64**2,2)
    # x=np.stack([np.sort(np.random.rand(64**2)),np.sort(np.random.rand(64**2))]).T
    xx,yy=np.meshgrid(np.linspace(0,1,64),np.linspace(0,1,64))
    x=np.stack([xx.flatten(),yy.flatten()]).T
    bsv=BSV(x,L=100,store_eig=True,basis_opt='wavelet', q=1.0) # constrast with q=2.0
    ## temporal class
    t=np.linspace(0,1,10)[:,np.newaxis]
    # x=np.random.rand(100,1) 
    epp=EPP(t,L=10,store_eig=True,ker_opt='matern',l=.1,nu=1.5,q=2)
    ## spatiotemporal class
    stbp=STBP(bsv,epp,store_eig=True)
    verbose=stbp.comm.rank==0 if stbp.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(stbp.eigv[:min(10,stbp.L)],4))
        print('Eigenvectors :', np.round(stbp.eigf[:,:min(10,stbp.L)],4))

    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))

    v=stbp.rnd(n=2)
    C=stbp.tomat()
    Cv=C.dot(v)
    Cv_te=stbp.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))

    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))

#     v=stbp.rnd(n=2)
#     invCv=spsla.spsolve(C,v)
# #     C_op=spsla.LinearOperator((stbp.N,)*2,matvec=lambda v:stbp.mult(v))
# #     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
#     invCv_te=stbp.act(v,-1)
#     if verbose:
#         print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))

    X=stbp.rnd(n=10)
    logpdf,_=stbp.logpdf(X)
    if verbose:
        print('Log-pdf of a matrix normal random variable: {:.4f}'.format(logpdf))
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))

    stbp2=stbp; stbp2.bsv.q=2; stbp2=stbp2.update(bsv=stbp2.bsv.update(l=stbp.bsv.l))
    u=stbp2.rnd()
    v=stbp2.rnd()
    h=1e-6
    dlogpdfv_fd=(stbp2.logpdf(u+h*v)[0]-stbp2.logpdf(u)[0])/h
    dlogpdfv=-stbp2.solve(u).T.dot(v)
    rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v) # this in general is not small because of C_t; try small correlation length in EPP, e.g. l=0.001 
    if verbose:
        print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
    if verbose:
        print('time: %.5f'% (time.time()-t3))
    
    import matplotlib.pyplot as plt
    
    u=stbp.rnd(n=5).reshape((stbp.I,stbp.J,-1),order='F')
    nrow=5; ncol=5
    fig, axes=plt.subplots(nrows=nrow,ncols=nrow,sharex=True,sharey=True,figsize=(15,12))
    for i in range(nrow):
        for j in range(ncol):
            ax=axes.flat[i*ncol+j]
            ax.imshow(u[:,j,i].reshape((int(np.sqrt(u.shape[0])),-1)),origin='lower')
            ax.set_aspect('auto')
            if i==0: ax.set_title('t='+str(j),fontsize=16)
    plt.show()