#!/usr/bin/env python
"""
Class definition of whitening non-Gaussian (STBP) distribution
-------------------------------------------------------------------------
Created February 1, 2023 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu lanzithinking@outlook.com"

import os
import numpy as np

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class whiten:
    """
    Whiten the non-Gaussian (STBP) prior
    """
    def __init__(self,prior):
        self.prior = prior # STBP prior containing bsv (spatial) and qep (temporal) submodules
        self.mean = None if self.prior.mean is None else self.stbp2wn(self.prior.mean)
    
    def wn2qep(self,z,dord=0,q=None):
        """
        White noise (z) representation of a q-EP random variable (xi), Lmd: z --> xi
        """
        if q is None: q = self.prior.qep.q
        _z = z.reshape((self.prior.L,self.prior.J,-1),order='F') # (L,J,_)
        nm_z = np.linalg.norm(_z,axis=1,keepdims=True)
        if dord==0:
            return self.prior.qep.act(_z*nm_z**(2/q-1),alpha=0.5,transp=True).squeeze()#,chol=False) # (L,J)
        if dord==1:
            def grad(v, adj=False):
                _v = v.reshape((self.prior.L,self.prior.J,-1),order='F')
                if adj:
                    _v = self.prior.qep.act(_v, alpha=0.5,transp=True,adjt=adj)#,chol=False)
                    dLmdv = _z*np.sum(_z*_v,axis=1,keepdims=True)*nm_z**(2/q-3)*(2/q-1) + _v*nm_z**(2/q-1)
                    return dLmdv.squeeze()
                else:
                    return self.prior.qep.act(_z*np.sum(_z*_v,axis=1,keepdims=True)*nm_z**(2/q-3)*(2/q-1) + _v*nm_z**(2/q-1), alpha=0.5,transp=True).squeeze()#,chol=False)
            return grad
        if dord==2:
            def hess(v, w, adj=False):
                _v = v.reshape((self.prior.L,self.prior.J,-1),order='F')
                _w = w.reshape((self.prior.L,self.prior.J,-1),order='F')
                Hv0 = (2/q-1)*self.prior.qep.act(_w*np.sum(_z*_v,axis=1,keepdims=True)*nm_z**(2/q-3), alpha=0.5,transp=True,adjt=adj)
                Hv1 = (2/q-1)*self.prior.qep.act(_z*nm_z**(2/q-3), alpha=0.5,transp=True)
                Hv2 = (2/q-1)*self.prior.qep.act(_z*np.sum(_z*_v,axis=1,keepdims=True)*nm_z**(2/q-5)*(2/q-3) + _v*nm_z**(2/q-3), alpha=0.5,transp=True)
                if adj:
                    wHv = Hv0 + np.sum(_w*Hv1,axis=1,keepdims=True)*_v + np.sum(_w*Hv2,axis=1,keepdims=True)*_z
                else:
                    wHv = Hv0 + np.sum(_w*_v,axis=1,keepdims=True)*Hv1 + np.sum(_w*_z,axis=1,keepdims=True)*Hv2
                return wHv.squeeze()
            return hess
    
    def wn2stbp(self,z,dord=0,q=None):
        """
        White noise (z) representation of a STBP random variable (u), T: z --> u
        """
        if q is None: q = self.prior.bsv.q
        if dord==0:
            return self.prior.C_act(self.wn2qep(z, dord), 1/q).squeeze()
        if dord==1:
            return lambda v,adj=False: self.wn2qep(z, dord)(self.prior.C_act(v, 1/q),adj=adj).reshape(v.shape,order='F') if adj else self.prior.C_act(self.wn2qep(z, dord)(v), 1/q).squeeze()
        if dord==2:
            return lambda v,w,adj=False: self.wn2qep(z, dord)(self.prior.C_act(v, 1/q), w,adj=adj).reshape(v.shape,order='F')
    
    def qep2wn(self,xi,dord=0,q=None):
        """
        Inverse white noise (z) representation of a q-EP random variable (xi), invLmd: xi --> z
        """
        if q is None: q = self.prior.qep.q
        _xi = self.prior.qep.act(xi.reshape((self.prior.L,self.prior.J,-1),order='F'),alpha=-0.5,transp=True) # (L,J,_)
        nm_xi = np.linalg.norm(_xi,axis=1,keepdims=True)
        if dord==0:
            return (_xi*nm_xi**(q/2-1)).squeeze()
        if dord==1:
            def grad(v, adj=False):
                _v = v.reshape((self.prior.L,self.prior.J,-1),order='F')
                if adj:
                    return self.prior.qep.act(_xi*np.sum(_xi*_v,axis=1,keepdims=True)*nm_xi**(q/2-3)*(q/2-1) + _v*nm_xi**(q/2-1), alpha=-0.5,transp=True,adjt=adj).squeeze()#,chol=False)
                else:
                    _v = self.prior.qep.act(_v, alpha=-0.5,transp=True)
                    diLmdv = _xi*np.sum(_xi*_v,axis=1,keepdims=True)*nm_xi**(q/2-3)*(q/2-1) + _v*nm_xi**(q/2-1)
                    return diLmdv.squeeze()
            return grad
    
    def stbp2wn(self,u,dord=0,q=None):
        """
        Inverse white noise (z) representation of a STBP random variable (u), invT: u --> z
        """
        if q is None: q = self.prior.bsv.q
        if dord==0:
            return self.qep2wn(self.prior.C_act(u, -1/q),dord).reshape(u.shape,order='F')
        if dord==1:
            return lambda v,adj=False: self.prior.C_act(self.qep2wn(self.prior.C_act(u, -1/q), dord)(v, adj=adj), -1/q).squeeze() #if adj else self.prior.C_act(self.qep2wn(self.prior.C_act(u, -1/q), dord)(v), -1/q).squeeze()
    
    def jacdet(self,z,dord=0,q=None):
        """
        (log) Jacobian determinant log dT = log dLmd + C
        """
        if q is None: q = self.prior.qep.q
        _z = z.reshape((self.prior.L,self.prior.J,-1),order='F') # (L,J,_)
        nm_z = np.linalg.norm(_z,axis=1,keepdims=True)
        if dord==0:
            return (2/q-1)*self.prior.J*np.log(nm_z).sum()
        if dord==1:
            return (2/q-1)*self.prior.J*(_z/nm_z**2).reshape(z.shape,order='F')
    
    def sample(self, mean=None):
        """
        Generate white noise sample
        """
        if mean is None:
            mean=self.mean
        u=np.random.randn(self.prior.dim)
        if mean is not None:
            u+=mean
        return u
    
if __name__ == '__main__':
    from emoji import emoji
    
    seed=2022
    np.random.seed(seed)
    # define emoji Bayesian inverse problem
    spat_args={'basis_opt':'Fourier','l':1,'s':1,'q':1.01,'L':2000}
    # spat_args={'basis_opt':'wavelet','wvlet_typ':'Meyer','l':1,'s':2,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'s':1.5,'q':1.0,'L':100}
    store_eig = True
    emj = emoji(spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed, init_param=True)
    emj.prior.mean = emj.init_parameter
    # define whitened object
    wht = whiten(emj.prior)
    
    # test
    h=1e-8; z, v, w=np.random.randn(3,emj.prior.L*emj.prior.J)
    # wn2qep (Lmd)
    print('**** Testing wn2qep (Lmd) ****')
    val,grad,hess=wht.wn2qep(z,0),wht.wn2qep(z,1),wht.wn2qep(z,2)
    val1,grad1=wht.wn2qep(z+h*v,0),wht.wn2qep(z+h*w,1)
    print('error in gradient: %0.8f' %(np.linalg.norm((val1-val)/h-grad(v))/np.linalg.norm(v)))
    print('error in Hessian: %0.8f' %(np.linalg.norm((grad1(v)-grad(v))/h-hess(v,w))/np.sqrt(np.linalg.norm(v)*np.linalg.norm(w))))
    # wn2stbp (T)
    print('\n**** Testing wn2stbp (T) ****')
    val,grad,hess=wht.wn2stbp(z,0),wht.wn2stbp(z,1),wht.wn2stbp(z,2)
    val1,grad1=wht.wn2stbp(z+h*v,0),wht.wn2stbp(z+h*w,1)
    print('error in gradient: %0.8f' %(np.linalg.norm((val1-val)/h-grad(v))/np.linalg.norm(v)))
    print('error in Hessian: %0.8f' %(np.linalg.norm((grad1(v)-grad(v))/h-hess(v,w))/np.sqrt(np.linalg.norm(v)*np.linalg.norm(w))))
    
    h=1e-8; xi, v=np.random.randn(2,emj.prior.L*emj.prior.J)
    # qep2wn (invLmd)
    print('\n**** Testing qep2wn (invLmd) ****')
    val,grad=wht.qep2wn(xi,0),wht.qep2wn(xi,1)
    val1=wht.qep2wn(xi+h*v,0)
    print('error in gradient: %0.8f' %(np.linalg.norm((val1-val)/h-grad(v))/np.linalg.norm(v)))
    xi1=wht.wn2qep(val,0).flatten(order='F')
    print('Relative error of Lmd-invLmd in a random direction between composition and identity: %.10f' % (np.linalg.norm(xi1-xi)/np.linalg.norm(xi)) )
    xi2=wht.qep2wn(wht.wn2qep(xi,0),0).flatten(order='F')
    print('Relative error of invLmd-Lmd in a random direction between composition and identity: %.10f' % (np.linalg.norm(xi2-xi)/np.linalg.norm(xi)))
    gradv=grad(v)
    v1=wht.wn2qep(val,1)(gradv).flatten(order='F')
    print('Relative error of dLmd-dinvLmd in a random direction between composition and identity: %.10f' % (np.linalg.norm(v1-v)/np.linalg.norm(v)))
    v2=wht.qep2wn(xi,1)(wht.wn2qep(val,1)(v)).flatten(order='F')
    print('Relative error of dinvLmd-dLmd in a random direction between composition and identity: %.10f' % (np.linalg.norm(v2-v)/np.linalg.norm(v)))
    
    h=1e-8; u, v=np.random.randn(2,emj.prior.L*emj.prior.J)
    # stbp2wn (invT)
    print('\n**** Testing stbp2wn (invT) ****')
    val,grad=wht.stbp2wn(u,0),wht.stbp2wn(u,1)
    val1=wht.stbp2wn(u+h*v,0)
    print('error in gradient: %0.8f' %(np.linalg.norm((val1-val)/h-grad(v))/np.linalg.norm(v)))
    u1=wht.wn2stbp(val,0)
    print('Relative error of T-invT in a random direction between composition and identity: %.10f' % (np.linalg.norm(u1-u)/np.linalg.norm(u)))
    u2=wht.stbp2wn(wht.wn2stbp(u,0),0)
    print('Relative error of invT-T in a random direction between composition and identity: %.10f' % (np.linalg.norm(u2-u)/np.linalg.norm(u)))
    gradv=grad(v)
    v1=wht.wn2stbp(val,1)(gradv)
    print('Relative error of dT-dinvT in a random direction between composition and identity: %.10f' % (np.linalg.norm(v1-v)/np.linalg.norm(v)))
    v2=wht.stbp2wn(u,1)(wht.wn2stbp(val,1)(v))
    print('Relative error of dinvT-dT in a random direction between composition and identity: %.10f' % (np.linalg.norm(v2-v)/np.linalg.norm(v)))
    
    h=1e-8; z, v=np.random.randn(2,emj.prior.L*emj.prior.J)
    # jacdet
    print('\n**** Testing jacdet ****')
    val,grad=wht.jacdet(z,0),wht.jacdet(z,1)
    val1=wht.jacdet(z+h*v,0)
    print('error in gradient of Jacobian determinant: %0.8f' %(np.linalg.norm((val1-val)/h-grad.dot(v))/np.linalg.norm(v)))