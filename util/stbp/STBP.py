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
__version__ = "1.0"
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
from util.stbp.qEP import *
from util.stbp.linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class STBP(BSV):
    def __init__(self,spat,temp,store_eig=False,**kwargs):
        """
        Initialize the STBP class with spatial (BSV) class bsv, temporal (qEP) class qep and the dynamic eigenvalues Lambda
        spat: spatial class (discrete size I x L)
        temp: temporal class (discrete size J x L)
        gamma: decaying eigenvalues
        store_eig: indicator to store eigen-pairs, default to be false
        spdapx: use speed-up or approximation
        -----------------------------------------------
        u(x,t) = sum_{l=1}^infty lambda_l(t) * phi_l(x)
        lambda_l = gamma_l xi_l(t) ~ qEP(0,cov_l), xi_l ~ qEP(0, C)
        phi_l - basis of Besov prior
        """
        if type(spat) is BSV:
            self.bsv=spat # spatial class
        else:
            self.bsv=BSV(spat,store_eig=store_eig,**kwargs.pop('spat_args',{}))
        if type(temp) is qEP:
            self.qep=temp # temporal class
        else:
            self.qep=qEP(temp,store_eig=store_eig or Lambda is None,**kwargs.pop('temp_args',{}))
        self.parameters=kwargs # all parameters of the kernel
        self.I,self.J=self.bsv.N,self.qep.N # spatial and temporal dimensions
        self.N=self.I*self.J # joint dimension (number of total inputs per trial)
        self.L=self.bsv.L#*self.qep.L
        if self.L>self.I:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed the size of spatial basis!")
            self.L=self.I
        self.gamma=self.bsv._qrteigv(self.L)
        try:
            self.comm=MPI.COMM_WORLD
        except:
            print('Parallel environment not found. It may run slowly in serial.')
            self.comm=None
        self.spdapx=self.parameters.get('spdapx',self.N>1e3)
        self.store_eig=store_eig
        # self.store_eig=False # Use BSV and qEP eigen-decomposition, not its own!
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self.eigs(**kwargs)
    
    def tomat(self,**kwargs):
        """
        Get kernel as matrix
        """
        alpha=kwargs.get('alpha',1)
        C = [self.bsv.tomat(alpha=alpha),self.qep.tomat(alpha=alpha)] # (IxI,JxJ)
        # if self.spdapx and not sps.issparse(C):
        #     warnings.warn('Possible memory overflow!')
        return C
    
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
            if v.shape[0]!=self.I:
                v=v.reshape((self.I,self.J,-1),order='F')
            if np.ndim(v)<3: v=v[:,:,None]
            Cv=self.qep.act(self.bsv.mult(v,alpha=alpha),alpha=alpha,transp=True).reshape((self.N,-1),order='F') # (IJ,K_)
        return Cv
    
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
            rtL=np.sqrt(L)
            rtL=int(rtL)+(not rtL.is_integer())
            lambda_x,Phi_x=self.bsv.eigs(rtL); lambda_t,Phi_t=self.qep.eigs(rtL)
            if not np.sqrt(L).is_integer():
                if lambda_x[-1]<lambda_t[-1]:
                    lambda_x=lambda_x[:-1]; Phi_x=Phi_x[:,:-1]
                else:
                    lambda_t=lambda_t[:-1]; Phi_t=Phi_t[:,:-1]
            eigv=[lambda_x,lambda_t]; eigf=[Phi_x,Phi_t]
        else:
            eigv,eigf=self.eigv,self.eigf
            if L<self.L:
                rtL=np.sqrt(L)
                rtL=int(rtL)+(not rtL.is_integer())
                eigv = [eig[:rtL] for eig in eigv]; eigf = [eig[:,rtL] for eig in eigf]
                if not np.sqrt(L).is_integer():
                    if eigv[0][-1]<eigv[1][-1]:
                        eigv[0]=eigv[0][:-1]; eigf[0]=eigf[0][:,:-1]
                    else:
                        eigv[1]=eigv[1][:-1]; eigf[1]=eigf[1][:,:-1]
        return eigv,eigf
    
    def act(self,x,alpha=1,**kwargs):
        """
        Obtain the action of C^alpha
        y=C^alpha *x
        """
        return BSV.act(self,x,alpha=alpha,**kwargs)
    
    def update(self,bsv=None,qep=None):
        """
        Update the eigen-basis
        """
        if bsv is not None:
            self.bsv=bsv; self.I=self.bsv.N; self.N=self.I*self.J
        if qep is not None:
            self.qep=qep; self.J=self.qep.N; self.N=self.I*self.J
        if self.store_eig:
            self.eigv,self.eigf=self.eigs(upd=True)
        return self
    
    def logdet(self):
        """
        Compute log-determinant of the kernel C: log|C|
        """
        return BSV.logdet(self)
    
    def logpdf(self,X,incldet=True):
        """
        Compute logpdf of centered spatiotemporal Besov process X ~ STBP(0,C,q)
        """
        if X.shape[:2]!=(self.I,self.J): X=X.reshape((self.I,self.J,-1),order='F')
        if np.ndim(X)<3: X=X[:,:,None]
        if not self.spdapx:
            # logpdf,q_ldet=BSV.logpdf(self.bsv,self.qep.act(X,alpha=-.5,transp=True).reshape((self.I,-1),order='F'),incldet=incldet)
            logpdf,q_ldet=self.qep.logpdf(self.bsv.act(X,alpha=-1.0/self.bsv.q).swapaxes(0,1).reshape((self.J,-1),order='F'),incldet=incldet) # works for single trial
        else:
            eigv,eigf=self.bsv.eigs();
            abs_eigv=abs(eigv)
            # q_ldet=-X.shape[1]*np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(float).eps]))*self.J if incldet else 0
            # proj_X=eigf.T.dot(X.reshape((self.J,self.I,-1)))/self.gamma[:,None,None] # (L,J,K_)
            q_ldet=-X.shape[2]*np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(float).eps]))/self.bsv.q if incldet else 0
            proj_X=np.tensordot(eigf,X,axes=(0,0))/eigv[:,None,None]**(1.0/self.bsv.q) # (L,J,K_)
            proj_X=proj_X.swapaxes(0,1).reshape((self.J,-1),order='F')
            qep_norm=self.qep.logpdf(proj_X,out='norms')
            qsum=-0.5*np.sum(qep_norm**(self.bsv.q/self.qep.q))
            logpdf=q_ldet+qsum
        return logpdf,q_ldet
    
    def rnd(self,n=1):
        """
        Generate spatiotemporal Besov random function (vector) rv ~ STBP(0,C,q)
        """
        qep_rv=self.qep.rnd(n=self.L*n).reshape((-1,self.L,n),order='F')*self.gamma[None,:,None] # (J,L,n)
        _,Phi_x=self.bsv.eigs();
        rv=Phi_x.dot(qep_rv).reshape((-1,n),order='F') # (N,n)
        return rv

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    # L=20
    ## spatial class
    # x=np.random.rand(64**2,2)
    # x=np.stack([np.sort(np.random.rand(64**2)),np.sort(np.random.rand(64**2))]).T
    xx,yy=np.meshgrid(np.linspace(0,1,16),np.linspace(0,1,16))
    x=np.stack([xx.flatten(),yy.flatten()]).T
    bsv=BSV(x,L=100,store_eig=True,basis_opt='wavelet', q=1.0) # constrast with q=2.0
    ## temporal class
    t=np.linspace(0,1,10)[:,np.newaxis]
    # x=np.random.rand(100,1) 
    qep=qEP(t,L=10,store_eig=True,ker_opt='matern',l=.1,nu=1.5,q=2)
    ## spatiotemporal class
    stbp=STBP(bsv,qep,store_eig=True)
    verbose=stbp.comm.rank==0 if stbp.comm is not None else True
    # if verbose:
    #     print('Eigenvalues :', np.round(stbp.eigv[:min(10,stbp.L)],4))
    #     print('Eigenvectors :', np.round(stbp.eigf[:,:min(10,stbp.L)],4))

    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))

    u_samp=stbp.rnd(n=5).reshape((stbp.I,stbp.J,-1),order='F')
    v=stbp.rnd(n=2).reshape((stbp.I,stbp.J,-1),order='F')
    C=stbp.tomat()
    Cv=multf(C[1],multf(C[0],v),transp=True).reshape((stbp.N,-1),order='F')
    Cv_te=stbp.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))

    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))

    # v=stbp.rnd(n=2)
    # invCv=spsla.spsolve(C,v)
    invCv=mdivf(C[1],mdivf(C[0],v),transp=True).reshape((stbp.N,-1),order='F')
#     C_op=spsla.LinearOperator((stbp.N,)*2,matvec=lambda v:stbp.mult(v))
#     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
    invCv_te=stbp.act(v,-1)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))

    X=stbp.rnd(n=10)
    logpdf,_=stbp.logpdf(X)
    if verbose:
        print('Log-pdf of a STBP random variable: {:.4f}'.format(logpdf))
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))

    stbp.bsv.q=2
    u=stbp.rnd()
    v=stbp.rnd()
    h=1e-7
    dlogpdfv_fd=(stbp.logpdf(u+h*v,incldet=False)[0]-stbp.logpdf(u,incldet=False)[0])/h
    dlogpdfv=-stbp.solve(u).T.dot(v)
    rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v) # this in general is not small because of C_t; try small correlation length in qEP, e.g. l=1e-8 
    if verbose:
        print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
    if verbose:
        print('time: %.5f'% (time.time()-t3))
    
    import matplotlib.pyplot as plt
    
    nrow=5; ncol=5
    fig, axes=plt.subplots(nrows=nrow,ncols=nrow,sharex=True,sharey=True,figsize=(15,12))
    for i in range(nrow):
        for j in range(ncol):
            ax=axes.flat[i*ncol+j]
            ax.imshow(u_samp[:,j,i].reshape((int(np.sqrt(u_samp.shape[0])),-1)),origin='lower')
            ax.set_aspect('auto')
            if i==0: ax.set_title('t='+str(j),fontsize=16)
    plt.show()