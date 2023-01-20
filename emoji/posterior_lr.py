#!/usr/bin/env python
"""
Class definition of Gaussian approximation for the posterior measure N(nu,K) with mean function mu and covariance operator K
where K^(-1) = I^(-1) + H(u), with H(u) being Hessian (or its Gaussian-Newton approximation) of misfit; and the prior N(mu, I)
---- low rank approximation ----
-------------------------------------------------------------------------
Created January 11, 2023 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.4"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu lanzithinking@outlook.com"

import os
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# self defined modules
import sys
sys.path.append( "../" )
from util.stbp.linalg import *
from util.Eigen import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class posterior:
    """
    Gaussian approximation of the posterior measure
    """
    def __init__(self,H,N=None,L=None,store_eig=False,**kwargs):
        self.H = H
        self.N = N if N is not None else self.H.shape[0]
        if L is None:
            L=min(self.N,100)
        self.L=L # truncation in Karhunen-Loeve expansion
        if self.L>self.N:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed size of the discrete basis!")
            self.L=self.N
        self.spdapx=kwargs.get('spdapx',self.N>1e3)
        self.store_eig=store_eig
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self.eigs(**kwargs)
    
    def mult(self,v,**kwargs):
        """
        Kernel multiply a function (vector): K*v
        """
        return itsol(self.H,v,solver=kwargs.pop('solver','cgs'))
    
    def solve(self,v):
        """
        Kernel solve a function (vector): K^(-1)*v
        """
        # return self.H.dot(v)
        return self.H(v)
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis of H
        H * eigf_i = eigf_i * eigv_i, i=1,...,L
        """
        if L is None:
            L=self.L;
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            # import time
            if not self.spdapx:
                maxiter=kwargs.pop('maxiter',100)
                tol=kwargs.pop('tol',1e-10)
                try:
                    # start = time.time()
                    H_op=self.H if isinstance(self.H,spsla.LinearOperator) else spsla.LinearOperator((self.N,)*2,self.H)
                    eigv,eigf=spsla.eigs(H_op,min(L,self.N-1),maxiter=maxiter,tol=tol)#,which='SM')
                    # end = time.time()
                    # print('Time used is %.4f' % (end-start))
                except Exception as divg:
                    print(*divg.args)
                    eigv,eigf=divg.eigenvalues,divg.eigenvectors
                eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                # eigv=pow(eigv,-1); #eigf=eigf[:,::-1]
                eigv=np.pad(eigv,(0,L-len(eigv)),mode='constant'); eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
            else:
                # start = time.time()
                eigv,eigf=eigen_RA(self.H,dim=self.N,k=L)#,which='SM')
                # end = time.time()
                # print('Time used is %.4f' % (end-start))
                eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                # eigv=pow(eigv,-1); #eigf=eigf[:,::-1]
        else:
            eigv,eigf=self.eigv,self.eigf
            eigv=eigv[:L]; eigf=eigf[:,:L]
        return eigv,eigf
    
    def sample(self, mean=None):
        """
        Sample a random function u ~ N(0, K)
        """
        # u=self.K_act(np.random.randn(self.N),comp=0.5)
        eigv, eigf=self.eigs()
        u=eigf.dot(np.random.randn(len(eigv))/np.sqrt(eigv+1))
        if mean is not None:
            u+=mean
        return u
    
    def K_act(self,u,comp=1,**kwargs):
        """
        Calculate operation of K^comp on vector u: u --> K^comp * u
        """
        if comp==0:
            return u
        else:
            if not self.spdapx:
                Ku=self.mult(u,**kwargs) if comp==1 else self.solve(u) if comp==-1 else None
            else:
                eigv, eigf=self.eigs()
                if comp==1:
                    Ku=u-np.tensordot(eigf*eigv/(eigv+1),np.tensordot(eigf,u,axes=(0,0)),axes=1)
                elif comp==-1:
                    Ku=u+np.tensordot(eigf*eigv,np.tensordot(eigf,u,axes=(0,0)),axes=1)
                elif abs(comp)==0.5:
                    Ku=np.tensordot(eigf*pow(eigv+1,-comp),np.tensordot(eigf,u,axes=(0,0)),axes=1)
                else:
                    raise NotImplementedError('Action not defined!')
        return Ku
    
    def logdet(self,eigv=None):
        """
        Compute log-determinant of the kernel K: log|K|
        """
        if eigv is None:
            eigv,_=self.eigs()
        abs_eigv=abs(eigv+1)
        ldet=-np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(float).eps]))
        return ldet
    
# if __name__ == '__main__':
    # np.random.seed(2022)
    