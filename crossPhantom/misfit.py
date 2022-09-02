#!/usr/bin/env python
"""
Class definition of data-misfit for the dynamic linear example of crossPhantom.
------------------------------------------------------------------------
Created June 27, 2022 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__credits__ = "Mirjeta Pasha"
__license__ = "GPL"
__version__ = "0.4"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import scipy.sparse as sps
# import scipy.linalg as spla
import scipy.io as spio
import scipy.sparse.linalg as spsla
# import h5py # needed to unpack downloaded crossPhantom data (in .mat format)
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
        self.nzlvl = kwargs.pop('nzlvl',1.) # noise level
        # self.jit = kwargs.pop('jit',1e-3) # jitter to the noise covariance
        # get observations
        self.obs, nzcov, self.sz_x, self.sz_t = self.get_obs(**kwargs)
        # self.nzcov = self.nzlvl * max(np.diag(nzcov)) * (nzcov + self.jit * sps.eye(nzcov.shape[0]))
        self.nzcov = self.nzlvl * max(np.diag(nzcov)) * ( sps.eye(nzcov.shape[0]))
    
    def _gen_crossPhantom(self):
        """
        Generate crossPhantom observations
        """
        if not os.path.exists('./data'): os.makedirs('./data')
        if not os.path.exists('./data/DataDynamic_128x15.mat'):
            import requests
            print("downloading...")
            r = requests.get('https://zenodo.org/record/1341457/files/DataDynamic_128x15.mat')
            with open('./data/DataDynamic_128x15.mat', "wb") as file:
                file.write(r.content)
            print("CrossPhantom data downloaded.")
        # with h5py.File('./data/DataDynamic_128x15.mat', 'r') as f:
        #     A = sps.csc_matrix((f["A"]["data"], f["A"]["ir"], f["A"]["jc"]))
        #     normA = np.array(f['normA'])
        #     sinogram = np.array(f['sinogram']).T
        f = spio.loadmat('./data/DataDynamic_128x15.mat')
        A = f['A']
        sinogram = f['sinogram']
    
        T = 16
        N = np.sqrt(A.shape[1] / T)
        [mm, nn] = sinogram.shape
        ind = []
        for ii in range(int(nn /3)): # every 3 of 30 angles for 33 seconds
            ind.extend( np.arange(0,mm) + (3*ii)*mm )
        m2 = sinogram[:, ::3]
        A_small = A[ind, :]
        b = m2
        nt = int(T)
        nx = int(N)
        ny = int(N)
        b = b.reshape(-1, 1, order='F').squeeze()
        AA = list(range(T))
        B = list(range(T))
        delta = 0 # no added noise for this dataset
        for ii in range(T):
            AA[ii] = A_small[ 700*(ii):700*(ii+1), 16384*ii:16384*(ii+1) ] # 217 projections of size 128x128 at each of 10 selected angles
            B[ii] = b[ 700*(ii) : 700*(ii+1) ]
        return A_small, b, AA, B, nx, ny, nt, delta
    
    def observe(self):
        """
        Observe image projections
        """
        A, b, AA, B, nx, ny, nt, delta = self._gen_crossPhantom()
        sz_x = (nx,ny) # I = np.prod(sz_x) = nx*ny
        sz_t = nt # J = sz_t
        ops_proj = AA; obs_proj = B
        nzcov = np.cov(np.stack(obs_proj), rowvar=False)
        return (ops_proj,obs_proj), nzcov, sz_x, sz_t
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='crossPhantom_obs'
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzcov=loaded['nzcov']; sz_x=loaded['sz_x']; sz_t=loaded['sz_t']
            print('Observation file '+obs_file_name+' has been read!')
        except Exception as e:
            print(e); pass
            obs, nzcov, sz_x, sz_t = self.observe()
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzcov=nzcov, sz_x=sz_x, sz_t=sz_t)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzcov, sz_x, sz_t
    
    def cost(self, u=None, obs=None):
        """
        Evaluate misfit function for given images (vector) u.
        """
        ops_proj, obs_proj = self.obs
        if obs is None:
            if u.shape[0]!=np.prod(self.sz_x):
                u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
            obs = np.stack([ops_proj[j].dot(u[:,j]) for j in range(self.sz_t)]).T
        # val = 0
        # for i in range(self.sz_t):
        #     dif_obs = ops_proj[i].dot(u[:,i]) - obs_proj[i]
        #     # val += 0.5*np.sum(dif_obs*spla.solve(self.nzcov,dif_obs,sym_pos=True))
        #     val += 0.5*np.sum(dif_obs*spsla.spsolve(self.nzcov,dif_obs))
        dif_obs = obs - np.stack(obs_proj).T
        val = .5*np.sum(dif_obs*spsla.spsolve(self.nzcov,dif_obs))
        return val
    
    def grad(self, u=None, obs=None):
        """
        Compute the gradient of misfit
        """
        ops_proj, obs_proj = self.obs
        if obs is None:
            if u.shape[0]!=np.prod(self.sz_x):
                u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
            obs = np.stack([ops_proj[j].dot(u[:,j]) for j in range(self.sz_t)]).T
        
        dif_obs = obs - np.stack(obs_proj).T
        g = []
        for i in range(self.sz_t):
            # dif_obs = ops_proj[i].dot(u[:,i]) - obs_proj[i]
            # g.append( ops_proj[i].T.dot(spla.solve(self.nzcov,dif_obs,sym_pos=True)) )
            g.append( ops_proj[i].T.dot(spsla.spsolve(self.nzcov,dif_obs[:,i])) )
        return np.stack(g).T # (I,J)
    
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
    
    def reconstruct_anisoTV(self):
        """
        Reconstruct images by anisoTV
        """
        A, b, AA, B, nx, ny, nt, delta = self._gen_crossPhantom()
        L = self._anisoTV((nx,ny),nt)
        xhat = GKS(A, b, L, 1, 5, 0, 0)
        xx = np.reshape(xhat, (128,128,16), order="F")
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
        if np.ndim(rcstr_imgs)!=3: rcstr_imgs=rcstr_imgs.reshape(np.append(self.sz_x,self.sz_t),order='F')
        # plot
        import matplotlib.pyplot as plt
        plt.set_cmap('Greys')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        for i in range(rcstr_imgs.shape[2]):
            plt.imshow(rcstr_imgs[:,:,i])
            plt.title('t = '+str(i),fontsize=16)
            if save_imgs: plt.savefig(save_path+'/crossPhantom_'+str(i).zfill(len(str(rcstr_imgs.shape[2])))+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()
    
if __name__ == '__main__':
    np.random.seed(2022)
    import time
    from prior import *
    
    # define the misfit
    msft = misfit()
    # define the prior
    pri = prior(sz_x=msft.sz_x,sz_t=msft.sz_t)
    
    t0=time.time()
    # generate sample
    # u=np.random.rand(np.prod(msft.sz_x),msft.sz_t)
    u=pri.sample('fun').reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    nll=msft.cost(u)
    grad=msft.grad(u)
    print('The negative logarithm of likelihood at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))
    # test
    # v=np.random.rand(np.prod(msft.sz_x),msft.sz_t)
    v=pri.sample('fun').reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    h=1e-8
    gradv_fd=(msft.cost(u+h*v)-nll)/h
    gradv=np.sum(grad*v)
    rdiff_gradv=np.abs(gradv_fd-gradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
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
    # x_hat=msft.reconstruct_lse(lmda=10)
    # # plot
    # # import matplotlib.pyplot as plt
    # msft.plot_reconstruction(x_hat, save_imgs=True, save_path='./reconstruction/LSE')
    #
    # # evaluate the likelihood at anisoTV reconstruction
    # u=x_hat.reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    # nll=msft.cost(u)
    # grad=msft.grad(u)
    # print('The negative logarithm of likelihood at LSE reconstruction is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))