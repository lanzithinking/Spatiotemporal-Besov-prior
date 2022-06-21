#!/usr/bin/env python
"""
generic exponential power process
-- with kernel choices 'powered exponential' and 'Matern class'
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ ASU, 2022
-------------------------------
Created June 9, 2022 @ ASU
-------------------------------
https://github.com/lanzithinking/Spatiotemporal-Besov-prior
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, STBP project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import scipy.spatial.distance as spsd
from scipy.special import gammaln
# self defined modules
import sys
sys.path.append( "../../" )
# from __init__ import *
from util.stbp.linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class EPP:
    def __init__(self,x,L=None,store_eig=False,**kwargs):
        """
        Initialize the EPP class with inputs and kernel settings
        x: inputs
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        ker_opt: kernel option, default to be 'powered exponential'
        dist_f: distance function, default to be 'minkowski'
        sigma2: magnitude, default to be 1
        l: correlation length, default to be 0.5
        s: smoothness, default to be 2
        nu: matern class order, default to be 0.5
        q: norm power, default to be 1
        jit: jittering term, default to be 1e-6
        # (dist_f,s)=('mahalanobis',vec) for anisotropic kernel
        spdapx: use speed-up or approximation
        """
        self.x=x # inputs
        if self.x.ndim==1: self.x=self.x[:,None]
        self.parameters=kwargs # all parameters of the kernel
        self.ker_opt=self.parameters.get('ker_opt','powexp') # kernel option
        self.dist_f=self.parameters.get('dist_f','minkowski') # distance function
        self.sigma2=self.parameters.get('sigma2',1) # magnitude
        self.l=self.parameters.get('l',0.5) # correlation length
        self.s=self.parameters.get('s',2) # smoothness
        self.nu=self.parameters.get('nu',0.5) # matern class order
        self.q=self.parameters.get('q',1) # norm power
        self.jit=self.parameters.get('jit',1e-6) # jitter
        self.N,self.d=self.x.shape # size and dimension
        if L is None:
            L=min(self.N,100)
        self.L=L # truncation in Karhunen-Loeve expansion
        if self.L>self.N:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed size of the discrete basis!")
            self.L=self.N
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
    
    def _powexp(self,*args):
        """
        Powered exponential kernel: C(x,y)=sigma2*exp(-.5*(||x-y||/l)^s)
        """
        if len(args)==1:
            C=spsd.squareform(np.exp(-.5*pow(spsd.pdist(args[0],self.dist_f,p=self.s)/self.l,self.s)))+(1.+self.jit)*np.eye(self.N)
        elif len(args)==2:
            C=np.exp(-.5*pow(spsd.cdist(args[0],args[1],self.dist_f,p=self.s)/self.l,self.s))
        else:
            print('Wrong number of inputs!')
            raise
        C*=self.sigma2
        return C
    
    def _matern(self,*args):
        """
        Matern class kernel: C(x,y)=2^(1-nu)/Gamma(nu)*(sqrt(2*nu)*(||x-y||/l)^s)^nu*K_nu(sqrt(2*nu)*(||x-y||/l)^s)
        """
        if len(args)==1:
            scal_dist=np.sqrt(2.*self.nu)*pow(spsd.pdist(args[0],self.dist_f,p=self.s)/self.l,self.s)
            C=pow(2.,1-self.nu)/sp.special.gamma(self.nu)*spsd.squareform(pow(scal_dist,self.nu)*sp.special.kv(self.nu,scal_dist))+(1.+self.jit)*np.eye(self.N)
        elif len(args)==2:
            scal_dist=np.sqrt(2*self.nu)*pow(spsd.cdist(args[0],args[1],self.dist_f,p=self.s)/self.l,self.s)
            C=pow(2.,1-self.nu)/sp.special.gamma(self.nu)*pow(scal_dist,self.nu)*sp.special.kv(self.nu,scal_dist)
            C[scal_dist==0]=1
        else:
            print('Wrong number of inputs!')
            raise
        C*=self.sigma2
        return C
    
    def tomat(self):
        """
        Get kernel as matrix
        """
        kerf=getattr(self,'_'+self.ker_opt) # obtain kernel function
        C=kerf(self.x)
        if type(C) is np.matrix:
            C=C.getA()
        if self.spdapx and not sps.issparse(C):
            warnings.warn('Possible memory overflow!')
        return C
    
    def mult(self,v,**kwargs):
        """
        Kernel multiply a function (vector): C*v
        """
        transp=kwargs.get('transp',False) # whether to transpose the result
        if not self.spdapx:
            Cv=multf(self.tomat(),v,transp)
        else:
            kerf=getattr(self,'_'+self.ker_opt) # obtain kernel function
            prun=kwargs.get('prun',True) and self.comm # control of parallel run
            if prun:
                try:
#                     import pydevd; pydevd.settrace()
                    nproc=self.comm.size; rank=self.comm.rank
                    if nproc==1: raise Exception('Only one process found!')
                    Cv_loc=multf(kerf(self.x[rank::nproc,:],self.x),v,transp)
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
#                 Cv=np.squeeze([multf(kerf(x_i[np.newaxis,:],self.x),v,transp) for x_i in self.x],1+transp)
                Cv=np.concatenate([multf(kerf(x_i[np.newaxis,:],self.x),v,transp) for x_i in self.x])
            if transp: Cv=Cv.swapaxes(0,1)
            Cv+=self.sigma2*self.jit*v
        return Cv
    
    def solve(self,v,**kwargs):
        """
        Kernel solve a function (vector): C^(-1)*v
        """
        transp=kwargs.pop('transp',False)
        if not self.spdapx:
            invCv=mdivf(self.tomat(),v,transp)
        else:
            C_op=spsla.LinearOperator((self.N,)*2,matvec=lambda v:self.mult(v,transp=transp,prun=True))
            nd_v=np.ndim(v)
            v=v.reshape(v.shape+(1,)*(3-nd_v),order='F')
            invCv=np.array([itsol(C_op,v[:,:,k],solver='cgs',transp=transp,comm=kwargs.pop('comm',None)) for k in np.arange(v.shape[2])])
            if nd_v==3:
                invCv=invCv.transpose((1,2,0))
            else:
                invCv=np.squeeze(invCv,axis=tuple(np.arange(0,nd_v-3,-1)))
#             if transp: Cv=Cv.swapaxes(0,1)
        return invCv
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis
        C * eigf_i = eigf_i * eigv_i, i=1,...,L
        """
        if L is None:
            L=self.L;
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            maxiter=kwargs.pop('maxiter',100)
            tol=kwargs.pop('tol',1e-10)
            C_op=self.tomat() if not self.spdapx else spsla.LinearOperator((self.N,)*2,matvec=lambda v:self.mult(v,**kwargs))
            try:
                eigv,eigf=spsla.eigsh(C_op,min(L,C_op.shape[0]),maxiter=maxiter,tol=tol)
            except Exception as divg:
                print(*divg.args)
                eigv,eigf=divg.eigenvalues,divg.eigenvectors
            eigv=abs(eigv[::-1]); eigf=eigf[:,::-1]
            eigv=np.pad(eigv,(0,L-len(eigv)),mode='constant'); eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
        else:
            eigv,eigf=self.eigv,self.eigf
            eigv=eigv[:L]; eigf=eigf[:,:L]
        return eigv,eigf
    
    def act(self,x,alpha=1,**kwargs):
        """
        Obtain the action of C^alpha
        y=C^alpha *x
        """
        transp=kwargs.get('transp',False)
        if alpha==0:
            y=x
        elif alpha==1:
            y=self.mult(x,**kwargs)
        elif alpha==-1:
            y=self.solve(x,**kwargs)
        else:
            eigv,eigf=self.eigs(**kwargs)
            if alpha<0: eigv[abs(eigv)<np.finfo(np.float).eps]=np.finfo(np.float).eps
            y=multf(eigf*pow(eigv,alpha),multf(eigf.T,x,transp),transp)
        return y
    
    def logdet(self):
        """
        Compute log-determinant of the kernel C: log|C|
        """
        eigv,_=self.eigs()
        abs_eigv=abs(eigv)
        ldet=np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(np.float).eps]))
        return ldet
    
    def logpdf(self,X,chol=True):
        """
        Compute logpdf of centered exponential power distribution X ~ EPD(0,C,q)
        """
        assert X.shape[0]==self.N, "Non-conforming size!"
        if not self.spdapx:
            if chol:
                try:
                    cholC,lower=spla.cho_factor(self.tomat())
                    half_ldet=-X.shape[1]*np.sum(np.log(np.diag(cholC)))
                    quad=X*spla.cho_solve((cholC,lower),X)
                except Exception as e:#spla.LinAlgError:
                    warnings.warn('Cholesky decomposition failed: '+str(e))
                    chol=False
                    pass
            if not chol:
                half_ldet=-X.shape[1]*self.logdet()/2
                quad=X*self.solve(X)
        else:
            eigv,eigf=self.eigs(); rteigv=np.sqrt(abs(eigv)+self.jit)#; rteigv[rteigv<self.jit]+=self.jit
            half_ldet=-X.shape[1]*np.sum(np.log(rteigv))
            half_quad=eigf.T.dot(X)/rteigv[:,None]
            quad=half_quad**2
        quad=-0.5*np.sum(quad)**(self.q/2)
        # scal_fctr=X.shape[1]*(np.log(self.N)+gammaln(self.N/2)-self.N/2*np.log(np.pi)-gammaln(1+self.N/self.q)-(1+self.N/self.q)*np.log(2))
        logpdf=half_ldet+quad#+scal_fctr
        return logpdf,half_ldet#,scal_fctr
    
    def update(self,sigma2=None,l=None):
        """
        Update the eigen-basis
        """
        if sigma2 is not None:
            sigma2_=self.sigma2
            self.sigma2=sigma2
            if self.store_eig:
                self.eigv*=self.sigma2/sigma2_
        if l is not None:
            self.l=l
            if self.store_eig:
                self.eigv,self.eigf=self.eigs(upd=True)
        return self
    
    def rnd(self,n=1,MU=None):
        """
        Generate exponential power random function (vector) rv ~ EPD(0,C,q)
        """
        uS_rv=np.random.randn(self.N,n) # (N,n)
        uS_rv/=np.linalg.norm(uS_rv,axis=0,keepdims=True)
        rv=np.random.gamma(shape=self.N/self.q,scale=2,size=n)**(1./self.q)*self.act(uS_rv,alpha=0.5)
        if MU is not None:
            rv+=MU
        return rv

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    x=np.linspace(0,1,100)[:,np.newaxis]
    # x=np.random.rand(100,1) 
    epp=EPP(x,L=50,store_eig=True,ker_opt='matern',l=.2,nu=1.5,q=1.5)
    verbose=epp.comm.rank==0 if epp.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(epp.eigv[:min(10,epp.L)],4))
        print('Eigenvectors :', np.round(epp.eigf[:,:min(10,epp.L)],4))
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    v=epp.rnd(n=2)
    C=epp.tomat()
    Cv=C.dot(v)
    Cv_te=epp.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))
    
    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))
    
    v=epp.rnd(n=2)
    invCv=np.linalg.solve(C,v)
#     C_op=spsla.LinearOperator((epp.N,)*2,matvec=lambda v:epp.mult(v))
#     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
    invCv_te=epp.act(v,-1)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))
    
#     X=epp.rnd(n=10)
#     X=X.reshape((X.shape[0],5,2),order='F')
#     logpdf,_=epp.logpdf(X)
#     if verbose:
#         print('Log-pdf of an exponential power random variable: {:.4f}'.format(logpdf))
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))
    
    epp2=epp; epp2.q=2; epp2=epp2.update(l=epp.l)
    u=epp2.rnd()
    v=epp2.rnd()
    h=1e-5
    dlogpdfv_fd=(epp2.logpdf(u+h*v)[0]-epp2.logpdf(u)[0])/h
    dlogpdfv=-epp2.solve(u).T.dot(v)
    rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v)
    if verbose:
        print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
    if verbose:
        print('time: %.5f'% (time.time()-t3))
    
    
    # plot some random samples
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,figsize=(12,5))
    axes[0].plot(epp.eigf[:,:3])
    u=epp.rnd(n=3)
    axes[1].plot(u)
    plt.show()