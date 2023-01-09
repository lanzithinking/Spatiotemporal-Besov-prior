#!/usr/bin/env python
"""
Class definition of data-misfit for the dynamic linear example of gelPhantom.
------------------------------------------------------------------------
Created October 10, 2022 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Mirjeta Pasha"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import scipy.sparse as sps
# import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import h5py # needed to unpack downloaded data (in .mat format)
# import scipy.io as spio
import pylops

from cil.framework import AcquisitionGeometry
from cil.processors import Slicer
from cil.plugins.astra import ProjectionOperator
from cil.utilities.display import show2D
from utilities_dynamic_ct import read_frames

# self defined modules
import os, sys
sys.path.append( "../" )
# from package.gks import GKS

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
        self.obs, nzvar, self.sz_x, self.sz_t, self.ig, self.ag, self.ig_small, self.ag_small, self.A_proj, self.data = self.get_obs(**kwargs)
        # self.nzcov = self.nzlvl * max(np.diag(nzcov)) * (nzcov + self.jit * sps.eye(nzcov.shape[0]))
        self.nzcov = self.nzlvl * max(nzvar) * ( sps.eye(nzvar.shape[0], format='csr'))
    
    def _gen_gelPhantom(self):
        """
        Generate gelPhantom observations
        """
        if not os.path.exists('./data'): os.makedirs('./data')
        if not os.path.exists('./data/GelPhantomData_b4.mat'):
            import requests
            print("downloading...")
            r = requests.get('https://zenodo.org/record/3696817/files/GelPhantomData_b4.mat')
            with open('./data/GelPhantomData_b4.mat', "wb") as file:
                file.write(r.content)
            print("GelPhantom data downloaded.")
        path = "./data/"
        data_mat = "GelPhantomData_b4"
        file_info = read_frames(path, data_mat)
        # file_info = h5py.File(path+data_mat+'.mat','r')
        # Get sinograms + metadata
        sinograms = file_info['sinograms']
        frames = sinograms.shape[0]
        angles = file_info['angles']
        distanceOriginDetector = file_info['distanceOriginDetector']
        distanceSourceOrigin = file_info['distanceSourceOrigin']
        # Correct the pixel size
        pixelSize = 2*file_info['pixelSize']
        numDetectors = file_info['numDetectors']
        ag = AcquisitionGeometry.create_Cone2D(source_position = [0, distanceSourceOrigin],
                                       detector_position = [0, -distanceOriginDetector])\
                                    .set_panel(numDetectors, pixelSize)\
                                    .set_channels(frames)\
                                    .set_angles(angles, angle_unit="radian")\
                                    .set_labels(['channel','angle', 'horizontal'])

        ig = ag.get_ImageGeometry()
        ig.voxel_num_x = 256
        ig.voxel_num_y = 256
        data = ag.allocate()
        for i in range(frames):
            data.fill(sinograms[i], channel = i) 
        step = 20;
        name_proj = "data_{}".format(int(360/step))
        data = Slicer(roi={'angle':(0,360,step)})(data)
        ag = data.geometry
        A = ProjectionOperator(ig, ag, 'cpu')
        n_t = frames
        n_x = ig.voxel_num_x
        n_y = ig.voxel_num_y
        # Generate the small data
        x0 = ig.allocate()
        x0_small = x0.get_slice(channel = 0)
        ag_small = data.get_slice(channel=0).geometry
        ig_small = x0_small.geometry
        A_small = ProjectionOperator(ig_small, ag_small, 'cpu')
        temp = A_small.direct(x0_small)
        AA = list(range(n_t))
        for ii in range(n_t):
            AA[ii] = A_small
        B = list(range(n_t))
        for i in range(frames):
            temp = ((data.array)[i, :, :]).flatten()
            B[i] = temp
        return A, data, AA, B, n_x, n_y, n_t, ig, ag, ig_small, ag_small
    
    def observe(self):
        """
        Observe image projections
        """
        A, b, AA, B, nx, ny, nt, geometry_x, geometry_b, geometry_x_small, geometry_b_small = self._gen_gelPhantom()
        sz_x = (nx,ny) # I = np.prod(sz_x) = nx*ny
        sz_t = nt # J = sz_t
        ops_proj = AA; obs_proj = B
        # nzcov = np.cov(np.stack(obs_proj), rowvar=False)
        nzvar = np.var(np.stack(obs_proj), axis=0)
        
        A_cil_forward = lambda x: self.ten_to_vec( A.direct( self.vec_to_ten(x, geometry_x)[0] ), geometry_b)[0]
        A_cil_backward = lambda b: self.ten_to_vec( A.adjoint( self.vec_to_ten(b, geometry_b)[0] ), geometry_x)[0]
        A_projection_operator = pylops.FunctionOperator(A_cil_forward, A_cil_backward, np.prod(geometry_b.shape), np.prod(geometry_x.shape) )
        
#         A_cil_forward_small = lambda x: mat_to_vec(AA{1}.direct( self.vec_to_mat(x, geometry_x_small)[0] ), geometry_b_small)[0]
#         A_cil_backward_small = lambda b: mat_to_vec( AA{1}.adjoint( self.vec_to_mat(b, geometry_b_small)[0] ), geometry_x_small)[0]
#         A_projection_operator_small = pylops.FunctionOperator(cil_forward_small, cil_backward_small, np.prod(geometry_b_small.shape), np.prod(geometry_x_small.shape) )
        
        return (ops_proj,obs_proj), nzvar, sz_x, sz_t, geometry_x, geometry_b, geometry_x_small, geometry_b_small, A_projection_operator, b
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs, nzvar, sz_x, sz_t, ig, ag, ig_small, ag_small, A_proj, b = self.observe()
        return obs, nzvar, sz_x, sz_t, ig, ag, ig_small, ag_small, A_proj, b
    
    def cost(self, u=None, obs=None):
        """
        Evaluate misfit function for given images (vector) u.
        """
        ops_proj, obs_proj = self.obs
        if obs is None:
            if u.shape[0]!=np.prod(self.sz_x):
                u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
            obs = np.stack([self.mat_to_vec(ops_proj[i].direct(self.vec_to_mat(u[:, i], self.ig_small)[0]), self.ag_small)[0] for i in range(self.sz_t)]).T
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
            obs = np.stack([self.mat_to_vec(ops_proj[i].direct(self.vec_to_mat(u[:, i], self.ig_small)[0]), self.ag_small)[0] for i in range(self.sz_t)]).T
        dif_obs = obs - np.stack(obs_proj).T
        g = []
        for i in range(self.sz_t):
            g.append(self.mat_to_vec(ops_proj[i].adjoint(self.vec_to_mat(spsla.spsolve(self.nzcov,dif_obs[:,i]), self.ag_small)[0]), self.ig_small)[0])
        return np.stack(g).T # (I,J)
    
    def Hess(self, u=None, obs=None):
        """
        Compute the Hessian action of misfit
        """
        ops_proj, obs_proj = self.obs
        def hess(v):
            if v.shape[0]!=np.prod(self.sz_x):
                v=v.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
            obs_v = np.stack([self.mat_to_vec(ops_proj[i].direct(self.vec_to_mat(v[:, i], self.ig_small)[0]), self.ag_small)[0] for i in range(self.sz_t)]).T
            return self.grad(obs=obs_v+np.stack(obs_proj).T)
        return hess
    
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
    
    def reconstruct_anisoTV(self, iter=10):
        """
        Reconstruct images by anisoTV
        """
        (ops_proj,obs_proj), nzvar, sz_x, sz_t, geometry_x, geometry_b, geometry_x_small, geometry_b_small, A_projection_operator, b  =  self.get_obs()
        data_f = b.array.reshape(-1,1)
        L = self._anisoTV(sz_x, sz_t)
        try:
            from package.gks import GKS
            (x, x_history, lambdah, lambda_history) = GKS(A_projection_operator, data_f,  L, 1, iter)
        except Exception as e:
            print(e)
            from gks_tools import GKS
            x = GKS(A_projection_operator, data_f,  L, 1, iter)
        xx = np.reshape(x, sz_x+(sz_t,), order="F")
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
   
    def plot_reconstruction(self, rcstr_imgs, save_imgs = True, save_path='./reconstruction'):
        """
        Plot the reconstruction.
        """
        if np.ndim(rcstr_imgs)!=3: rcstr_imgs=rcstr_imgs.reshape(np.append(self.sz_x,self.sz_t),order='F')
        # plot
        import matplotlib.pyplot as plt
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        for i in range(rcstr_imgs.shape[2]):
            show2D(rcstr_imgs[:, :, i], num_cols=4, origin="upper",fix_range=(0,0.065),
                    cmap="inferno", title="Time-frame {}".format(i), size=(25, 20))
            if save_imgs: plt.savefig(save_path+'/gelPhantom_'+str(i).zfill(len(str(rcstr_imgs.shape[2])))+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()
    
    def vec_to_mat(self, x, geometry):
        v = geometry.allocate()
        [n1, n2] = v.shape
        temp = x.reshape(n1, n2)
        v.fill(temp)
        return v, n1, n2

    def mat_to_vec(self, x, geometry):
        temp = geometry.allocate()
        [n1, n2] = temp.shape
        v = x.array.flatten()
        return v, n1, n2
    
    def vec_to_ten(self, x, geometry):
        v = geometry.allocate()
        [n1, n2, n3] = v.shape
        temp = x.reshape(n1, n2, n3)
        v.fill(temp)
        return v, n1, n2, n3
    
    def ten_to_vec(self, x, geometry):
        temp = geometry.allocate()
        [n1, n2, n3] = temp.shape
        v = x.array.flatten()
        return v, n1, n2, n3
    
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
    hess=msft.Hess(u)
    # test
    # v=np.random.rand(np.prod(msft.sz_x),msft.sz_t)
    v=pri.sample('fun').reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    h=1e-6
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