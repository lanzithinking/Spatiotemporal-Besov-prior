#!/usr/bin/env python
"""
Class definition of Gaussian approximation for the posterior measure N(nu,K) with mean function mu and covariance operator K
where K^(-1) = C^(-1) + H(u), with H(u) being Hessian (or its Gaussian-Newton approximation) of misfit; and the prior N(mu, C)
-------------------------------------------------------------------------
Created January 11, 2023 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.2"
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
    def __init__(self,invK,N=None,L=None,store_eig=False,**kwargs):
        self.invK = invK
        self.N = N if N is not None else self.invK.shape[0]
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
        return itsol(self.invK,v,solver=kwargs.pop('solver','cgs'))
    
    def solve(self,v):
        """
        Kernel solve a function (vector): K^(-1)*v
        """
        # return self.invK.dot(v)
        return self.invK(v)
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis
        K * eigf_i = eigf_i * eigv_i, i=1,...,L
        """
        if L is None:
            L=self.L;
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            import time
            if not self.spdapx:
                maxiter=kwargs.pop('maxiter',100)
                tol=kwargs.pop('tol',1e-10)
                try:
                    # start = time.time()
                    invK_op=self.invK if isinstance(self.invK,spsla.LinearOperator) else spsla.LinearOperator((self.N,)*2,self.invK)
                    eigv,eigf=spsla.eigs(invK_op,min(L,self.N-1),which='SM',maxiter=maxiter,tol=tol)
                    # end = time.time()
                    # print('Time used is %.4f' % (end-start))
                except Exception as divg:
                    print(*divg.args)
                    eigv,eigf=divg.eigenvalues,divg.eigenvectors
                eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                eigv=pow(eigv,-1)[::-1]; eigf=eigf[:,::-1]
                eigv=np.pad(eigv,(0,L-len(eigv)),mode='constant'); eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
            else:
                start = time.time()
                eigv,eigf=eigen_RA(self.invK,dim=self.N,k=L,which='SM')
                end = time.time()
                print('Time used is %.4f' % (end-start))
                eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                eigv=pow(eigv,-1)[::-1]; eigf=eigf[:,::-1]
        else:
            eigv,eigf=self.eigv,self.eigf
            eigv=eigv[:L]; eigf=eigf[:,:L]
        return eigv,eigf
    
    def sample(self, mean=None):
        """
        Sample a random function u ~ N(0, K)
        """
        u=self.K_act(np.random.randn(self.N),comp=0.5)
        if mean is not None:
            u+=mean
        return u
    
    def K_act(self,u,comp=1,**kwargs):
        """
        Calculate operation of K^comp on vector u: u --> K^comp * u
        """
        if comp==0:
            return u
        elif comp==1 and not self.spdapx:
            Ku=self.mult(u,**kwargs)
        elif comp==-1:
            Ku=self.solve(u)
        else:
            eigv, eigf=self.eigs()
            if comp<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
            # import time
            # t_start=time.time()
            # Ku0=multf(eigf*pow(eigv,comp),multf(eigf.T,u))
            # print('time: %.4f' % (time.time()-t_start))
            # t_start=time.time()
            Ku=np.tensordot(eigf*pow(eigv,comp),np.tensordot(eigf,u,axes=(0,0)),axes=1)
            # print('time: %.4f' % (time.time()-t_start))
            # print('error: %.4e' % (abs(Ku-Ku0).max()))
        return Ku
    
    def logdet(self):
        """
        Compute log-determinant of the kernel K: log|K|
        """
        eigv,_=self.eigs()
        abs_eigv=abs(eigv)
        ldet=np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(float).eps]))
        return ldet
    
# if __name__ == '__main__':
    # np.random.seed(2022)
    