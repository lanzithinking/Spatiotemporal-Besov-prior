#!/usr/bin/env python
"""
Class definition of data-misfit for the dynamic linear example of emoji.
------------------------------------------------------------------------
Created June 27, 2022 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__credits__ = "Mirjeta Pasha"
__license__ = "GPL"
__version__ = "0.8"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import scipy.sparse as sps
# import scipy.linalg as spla
import scipy.sparse.linalg as spsla
# import h5py # needed to unpack downloaded emoji data (in .mat format)
import os

# self defined modules
from gks_tools import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class misfit(object):
    """
    Class definition of data-misfit function
    """
    def __init__(self, **kwargs):
        """
        Initialize data-misfit class with information of observations.
        """
        # get observations
        self.truth_option = kwargs.pop('truth_option', 1)
        self.truth_name = {0:'step',1:'turning'}[self.truth_option]
        self.n_x, self.n_t = kwargs.pop('n_x',2**4), kwargs.pop('n_t',20)
        self.size = self.n_x* self.n_t
        self.data_set = kwargs.pop('data_set','simulation_{}'.format(self.n_x)) # data set
        self.nzlvl = kwargs.pop('nzlvl',0.01) # noise level
        self.obs, self.nzvar, self.truth = self.get_obs(**kwargs)

        # self.nzcov = self.nzlvl * max(np.diag(nzcov)) * (nzcov + self.jit * sps.eye(nzcov.shape[0]))
        #self.nzcov = self.nzlvl * max(np.diag(nzcov)) * ( sps.eye(nzcov.shape[0], format='csr'))
   
    def _truef(self, opt=None):
        """
        Truth process
        """
        if opt is None: opt=self.truth_option
        
        if opt==0:
            f=lambda ts: np.array([1*(t>=0 and t<=1) + 0.5*(t>1 and t<=1.5) + 2*(t>1.5 and t<=2) for t in ts])
        elif opt==1:
            f=lambda ts: np.array([1.5*t*(t>=0 and t<=1) + (3.5-2*t)*(t>1 and t<=1.5) + (3*t-4)*(t>1.5 and t<=2) for t in ts])
        else:
            raise ValueError('Wrong option for truth!')
            

        m = lambda x, t: f(x)*np.sin(2*np.pi*t)  #(n_x, n_t)
        
              
        return m
    
    def observe(self, opt=None):
        """
        Observe time series by adding noise
        """
        if opt is None: opt=self.truth_option
        x=np.linspace(-1,1,self.n_x).reshape(self.n_x, 1)
        t=np.linspace(0,2,self.n_t).reshape(1, self.n_t)
        # parameters setting
        #sigma2_n=1e-2 # noise variance
        #C_x=exp(-pdist2(x,x,'squaredeuclidean')./(2*l_x)); C_t=exp(-(t-t').^2./(2*l_t))
        
        obs_ts = self._truef(opt)(x,t).reshape((self.n_x*self.n_t,), order='F')
        truth = obs_ts
        nzstd = self.nzlvl*np.linalg.norm(obs_ts)
        return obs_ts, nzstd**2, truth
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_opt = kwargs.pop('opt',self.truth_option)
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='simulation_obs_'+{0:'step',1:'turning'}[obs_opt]+'_'+str(self.n_x)
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzvar=loaded['nzvar']; truth=loaded['truth']
            print('Observation file '+obs_file_name+' has been read!')
            if obs.size!=self.size:
                raise ValueError('Stored observations not match the requested size! Regenerating...')
        except Exception as e:
            print(e); pass
            obs, nzvar, truth = self.observe(opt=obs_opt)
            obs += np.sqrt(nzvar) * np.random.RandomState(kwargs.pop('rand_seed',2022)).randn(*obs.shape)
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzvar=nzvar, truth=truth)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzvar, truth
    
    def cost(self, u=None, obs=None):
        """
        Evaluate misfit function for given images (vector) u.
        """
        
        obs = u
        dif_obs = obs-self.obs #obs - np.stack(obs_proj).T
        val = 0.5*np.sum(dif_obs**2/self.nzvar) #.5*np.sum(dif_obs*spsla.spsolve(self.nzcov,dif_obs))
        
#        if u.shape[0]!=np.prod(self.n_x):
#            u=u.reshape((self.n_x,-1),order='F') # (I,J)
        
        return val
    
    def grad(self, u=None, obs=None):
        """
        Compute the gradient of misfit
        """
        obs = u
        dif_obs = obs-self.obs
        g = dif_obs/self.nzvar
        
        return g #np.stack(g).T # (I,J)
    
    def Hess(self, u=None, obs=None):
        """
        Compute the Hessian action of misfit
        """
        def hess(v):
            if v.ndim==1 or v.shape[0]!=self.size: v=v.reshape((self.size,-1))
            dif_obs = v
            Hv = dif_obs/(self.nzvar if self.nzvar.size==1 else self.nzvar[:,None])
            return Hv.squeeze()
        
        return hess
    
    # @staticmethod
    # def is_diag(a):
    #     diag_elem = a.diagonal().copy()
    #     np.fill_diagonal(a,0)
    #     out = (a==0).all()
    #     np.fill_diagonal(a,diag_elem)
    #     return out
    
    def noise(self):
        """
        Generate Gaussian random noise with data covariance
        """
        z = np.random.randn(self.nzcov.shape[0],self.sz_t)
        if np.all(self.nzcov==np.diag(self.nzcov.diagonal())): #is_diag(self.nzcov):
            nzrv = np.sqrt(self.nzcov.diagonal())[:,None]*z
        else:
            nzrv = spla.cholesky(self.nzcov, lower=True).dot(z)
        return nzrv
    
    def _anisoTV(self, sz_x=None, sz_t=None):
        """
        Anisotropic total variation operator
        """
        if sz_x is None: sz_x=self.sz_x
        if sz_t is None: sz_t=self.sz_t
        _op = lambda n: sps.eye(n)-sps.eye(m=n,n=n,k=-1)
        D_x = sps.vstack((sps.kron(sps.eye(sz_x[0]),_op(sz_x[0])),sps.kron(_op(sz_x[1]),sps.eye(sz_x[0]))))
        D_t = _op(sz_t)[:-1]
        L = sps.vstack((sps.kron(sps.eye(sz_t),D_x),sps.kron(D_t,sps.eye(np.prod(sz_x)))))
        return L
    
    def reconstruct_anisoTV(self, iter=5):
        """
        Reconstruct images by anisoTV
        """
        A, b, AA, B, nx, ny, nt, delta = self._gen_emoji()
        L = self._anisoTV((nx,ny),nt)
        xhat = GKS(A, b, L, 1, iter, 0, 0)
        xx = np.reshape(xhat, (nx,ny,nt), order="F")
        return xx
    
    def reconstruct_LSE(self,lmda=0):
        """
        Reconstruct images by least square estimate
        """
        ops_proj, obs_proj = self.obs
        x_hat = []
        for i in range(self.sz_t):
            # XX = ops_proj[i].T.dot(ops_proj[i]) + lmda * sps.eye(np.prod(self.sz_x))
            # x_hat.append(spsla.spsolve(XX, ops_proj[i].T.dot(obs_proj[i])))
            x_hat.append(spsla.lsqr(ops_proj[i],obs_proj[i],damp=lmda)[0])
        return np.stack(x_hat).T
    
    def plot_reconstruction(self, rcstr_imgs, save_imgs=False, save_path='./reconstruction'):
        """
        Plot the reconstruction.
        """
        if np.ndim(rcstr_imgs)!=2: rcstr_imgs=rcstr_imgs.reshape(np.append(self.n_x,self.n_t),order='F')
        # plot
        truth = self.truth.reshape(np.append(self.n_x,self.n_t),order='F')
        import matplotlib.pyplot as plt
        plt.set_cmap('Greys')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        for i in range(rcstr_imgs.shape[1]):
            plt.plot(rcstr_imgs[:,i], label='estimate')
            plt.plot(truth[:,i],color='red', label='truth')
            plt.legend()
            plt.title('t = '+str(i),fontsize=16)
            if save_imgs: plt.savefig(save_path+'/simulation_'+str(i).zfill(len(str(rcstr_imgs.shape[1])))+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()
    
if __name__ == '__main__':
    np.random.seed(2022)
    import time
    from prior import *
    
    # define the misfit
    n_x=2**4
    msft = misfit(truth_option=1, n_x=n_x)
    # define the prior
    pri = prior(sz_x=msft.sz_x,sz_t=msft.sz_t)
    
    t0=time.time()
    # generate sample
    # u=np.random.rand(np.prod(msft.sz_x),msft.sz_t)
    u=pri.sample('fun').reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    nll=msft.cost(u)
    grad=msft.grad(u)
    print('The negative logarithm of likelihood at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))
    hess=msft.Hess(u)
    # test
    # v=np.random.rand(np.prod(msft.sz_x),msft.sz_t)
    v=pri.sample('fun').reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    h=1e-8
    gradv_fd=(msft.cost(u+h*v)-nll)/h
    gradv=np.sum(grad*v)
    rdiff_gradv=np.abs(gradv_fd-gradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    hessv_fd=(msft.grad(u+h*v)-grad)/h
    hessv=hess(v)
    rdiff_hessv=np.linalg.norm(hessv_fd-hessv)/np.linalg.norm(v)
    print('Relative difference of Hessian-action in a direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
    t1=time.time()
    print('time: %.5f'% (t1-t0))
    
    # # reconstruct the images by anisoTV
    # xx=msft.reconstruct_anisoTV()
    # # plot
    # # import matplotlib.pyplot as plt
    # msft.plot_reconstruction(xx, save_imgs=True, save_path='./reconstruction/anisoTV')
    #
    # # evaluate the likelihood at anisoTV reconstruction
    # u=xx.reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    # nll=msft.cost(u)
    # grad=msft.grad(u)
    # print('The negative logarithm of likelihood at anisoTV reconstruction is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))
    
    # # reconstruct the images by LSE
    # x_hat=msft.reconstruct_LSE(lmda=10)
    # # plot
    # # import matplotlib.pyplot as plt
    # msft.plot_reconstruction(x_hat, save_imgs=True, save_path='./reconstruction/LSE')
    #
    # # evaluate the likelihood at anisoTV reconstruction
    # u=x_hat.reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    # nll=msft.cost(u)
    # grad=msft.grad(u)
    # print('The negative logarithm of likelihood at LSE reconstruction is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))