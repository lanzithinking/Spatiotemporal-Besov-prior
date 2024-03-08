#!/usr/bin/env python
"""
Class definition of data-misfit for the dynamic linear example of annulus.
-------------------------------------------------------------------------
Created March 5, 2024 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__credits__ = "Shuyi Li"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import scipy.sparse as sps
# import scipy.linalg as spla
import scipy.sparse.linalg as spsla
# import h5py # needed to unpack downloaded emoji data (in .mat format)
import os

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
        self.sz_x, self.sz_t = kwargs.pop('n_x',2**4), kwargs.pop('n_t',10)
        if not hasattr(self.sz_x, "__len__"): self.sz_x=(self.sz_x,)*2 # I = np.prod(sz_x)
        self.nzvar = kwargs.get('nzvar',.1**2)
        if not hasattr(self.nzvar, "__len__"): self.nzvar=np.repeat(self.nzvar, np.prod(self.sz_x))
        self.times = np.linspace(0,1,self.sz_t,endpoint=False)
        self.times += self.times[1]
        self.obs, self.nzvar, self.truth = self.get_obs(**kwargs)
        # self.nzcov = max(self.nzvar) * ( sps.eye(nzvar.shape[0], format='csr'))
        # self.nzcov = sps.spdiags(self.nzvar[None,:], 0, format='csr')
    
    def _truef(self, x, t):
        """
        Truth process: u(x, t) = t * delta(sin(|x|/pi <=t))
        """
        return t * (np.sin(np.pi * np.linalg.norm(x, axis=-1,keepdims=True))>=t)
    
    def observe(self, **kwargs):
        """
        Observe time series by adding noise
        """
        nzvar = kwargs.pop('nzvar',self.nzvar)
        if not hasattr(nzvar, "__len__"): nzvar=np.repeat(nzvar, np.prod(self.sz_x))
        xx,yy=np.meshgrid(np.linspace(-1,1,self.sz_x[0]),np.linspace(-1,1,self.sz_x[1]))
        x=np.stack([xx.flatten(),yy.flatten()]).T
        t=kwargs.pop('times',self.times)
        truth=self._truef(x, t) # (I,J)
        obs = truth+np.sqrt(nzvar)[:,None]*np.random.RandomState(seed=kwargs.pop('seed',2022)).randn(np.prod(self.sz_x), self.sz_t) # (I,J)
        truth = truth.reshape(np.append(self.sz_x,self.sz_t),order='F') # (n_x,n_y,J)
        return obs, nzvar, truth
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.path.join(os.getcwd(),'data'))
        os.makedirs(obs_file_loc, exist_ok=True)
        obs_file_name='simulation_obs_I{}_J{}'.format(np.prod(self.sz_x), self.sz_t)
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzvar=loaded['nzvar']; truth=loaded['truth']
            print('Observation file '+obs_file_name+' has been read!')
        except Exception as e:
            print(e); pass
            obs, nzvar, truth = self.observe(**kwargs)
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzvar=nzvar, truth=truth)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzvar, truth
    
    def cost(self, u=None, obs=None):
        """
        Evaluate misfit function for given images (vector) u.
        """
        if obs is None:
            if u.shape[0]!=np.prod(self.sz_x):
                u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
            obs = u
        dif_obs = obs - self.obs
        # val = .5*np.sum(dif_obs*spsla.spsolve(self.nzcov,dif_obs))
        val = 0.5*np.sum(dif_obs**2/(self.nzvar if self.nzvar.size==1 else self.nzvar[:,None]))
        return val
    
    def grad(self, u=None, obs=None):
        """
        Compute the gradient of misfit
        """
        if obs is None:
            if u.shape[0]!=np.prod(self.sz_x):
                u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
            obs = u
        dif_obs = obs - self.obs
        # g = spsla.spsolve(self.nzcov,dif_obs)
        g = dif_obs/(self.nzvar if self.nzvar.size==1 else self.nzvar[:,None])
        return g # (I,J)
    
    def Hess(self, u=None, obs=None):
        """
        Compute the Hessian action of misfit
        """
        def hess(v):
            if v.shape[:2]!=(np.prod(self.sz_x),self.sz_t):
                v=v.reshape((np.prod(self.sz_x),self.sz_t,-1),order='F') # (I,J,K)
            if v.ndim==2: v=v[:,:,None]
            # Hv = np.stack([spsla.spsolve(self.nzcov,v[:,j,:]) for j in range(self.sz_t)]).swapaxes(0,1)
            Hv = v/(self.nzvar if self.nzvar.size==1 else self.nzvar[:,None,None])
            return Hv.squeeze()
        return hess # (I,J,K)
    
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
    
    def reconstruct_LSE(self,lmda=0):
        """
        Reconstruct images by least square estimate
        """
        obs = self.obs
        x_hat = []
        for j in range(self.sz_t):
            x_hat.append(spsla.lsqr(sps.eye(np.prod(self.sz_x)),obs[:,j],damp=lmda)[0])
        return np.stack(x_hat).T
    
    def plot_reconstruction(self, rcstr_imgs, save_imgs=False, save_path='./reconstruction', **kwargs):
        """
        Plot the reconstruction.
        """
        if np.ndim(rcstr_imgs)!=3: rcstr_imgs=rcstr_imgs.reshape(np.append(self.sz_x,self.sz_t),order='F')
        # plot
        import matplotlib.pyplot as plt
        plt.set_cmap(kwargs.pop('cmap','gray'))
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        for i in range(rcstr_imgs.shape[2]):
            plt.imshow(rcstr_imgs[:,:,i])
            # plt.axis('off')
            plt.xticks([]),plt.yticks([])
            if kwargs.get('time_label',True): plt.title('t = '+"{:.2f}".format(self.times[i]),fontsize=20)
            if save_imgs: plt.savefig(save_path+'/simulation_'+str(i).zfill(len(str(rcstr_imgs.shape[2])))+'.png',bbox_inches='tight')
            if kwargs.get('anim',True):
                plt.pause(.1)
                plt.draw()
    
if __name__ == '__main__':
    np.random.seed(2022)
    import time
    from prior import *
    
    # define the misfit
    n_x=2**8; n_t=100
    msft = misfit(n_x=n_x, n_t=n_t)
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
    h=1e-7
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
    
    # plot the true images
    msft.plot_reconstruction(msft.truth, save_imgs=True, save_path='./data/truth_simulation', time_label=False)
    # plot the observed images
    msft.plot_reconstruction(msft.obs, save_imgs=True, save_path='./data/obs_simulation', time_label=False)
    
    # # reconstruct the images by LSE
    # x_hat=msft.reconstruct_LSE(lmda=1)
    # # plot
    # # import matplotlib.pyplot as plt
    # msft.plot_reconstruction(x_hat, save_imgs=True, save_path='./reconstruction/LSE')
    #
    # # evaluate the likelihood at anisoTV reconstruction
    # u=x_hat.reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    # nll=msft.cost(u)
    # grad=msft.grad(u)
    # print('The negative logarithm of likelihood at LSE reconstruction is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))