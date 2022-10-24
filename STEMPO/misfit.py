#!/usr/bin/env python
"""
Class definition of data-misfit for the dynamic linear example of STEMPO.
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
# import h5py # needed to unpack downloaded data (in .mat format)
import scipy.io as spio
import astra

# self defined modules
# from gks_tools import *
import os,sys
sys.path.append( "../" )
# from package.gks import *
# from package.io import *
# from package.operators import *

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
        self.data_src = kwargs.pop('data_src','simulation') # data source
        self.obs, nzvar, self.sz_x, self.sz_t, self.truth = self.get_obs(**kwargs)
        # self.nzcov = self.nzlvl * max(np.diag(nzcov)) * (nzcov + self.jit * sps.eye(nzcov.shape[0]))
        self.nzcov = self.nzlvl * max(nzvar) * ( sps.eye(nzvar.shape[0]))
    
    def _gen_stempo(self, source=None):
        """
        Generate stempo observations
        """
        if source is None: source = self.data_src
        data_file = {'simulation':'stempo_ground_truth_2d_b4','real':'stempo_seq8x45_2d_b16'}[source]+'.mat'
        if not os.path.exists('./data'): os.makedirs('./data')
        if not os.path.exists('./data/'+data_file):
            import requests
            print("downloading...")
            r = requests.get('https://zenodo.org/record/7147139/files/'+data_file)
            with open('./data/'+data_file, "wb") as file:
                file.write(r.content)
            print("Stempo data downloaded.")
        if source=='simulation':
            truth = spio.loadmat('./data/'+data_file)
            image = truth['obj']
            nx, ny, nt = 560, 560, 20;
            anglecount = 10
            rowshift = 5
            columnsshift = 14
            nt = 20
            angleVector = list(range(nt))
            for t in range(nt):
                angleVector[t] = np.linspace(rowshift*t, 14*anglecount+ rowshift*t, num = anglecount+1)
            angleVectorRad = np.deg2rad(angleVector)
                    # Generate matrix versions of the operators and a large bidiagonal sparse matrix
            N = nx         # object size N-by-N pixels
            p = int(np.sqrt(2)*N)    # number of detector pixels
            # view angles
            theta = angleVectorRad#[0]#np.linspace(0, 2*np.pi, q, endpoint=False)   # in rad
            q = theta.shape[1]          # number of projection angles
            source_origin = 3*N                     # source origin distance [cm]
            detector_origin = N                       # origin detector distance [cm]
            detector_pixel_size = (source_origin + detector_origin)/source_origin
            detector_length = detector_pixel_size*p 
            saveA = list(range(nt))
            saveb = np.zeros((p*q, nt))
            saveb_true = np.zeros((p*q, nt))
            savee = np.zeros((p*q, nt))
            savedelta = np.zeros((nt, 1))
            savex_true = np.zeros((nx*ny, nt))
            B = list(range(nt))
            count = np.int_(360/nt)
            for i in range(nt):
                proj_geom = astra.create_proj_geom('fanflat', detector_pixel_size, p, theta[i], source_origin, detector_origin)
                vol_geom = astra.create_vol_geom(N, N)
                proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
                mat_id = astra.projector.matrix(proj_id)
                A_n = astra.matrix.get(mat_id)
                x_true = image[:, :, count*i]
                x_truef_sino = x_true.flatten(order='F') 
                savex_true[:, i] = x_truef_sino
                sn = A_n*x_truef_sino
                b_i = sn.flatten(order='F') 
                sigma = 0.01 # noise level
                e = np.random.normal(0, 1, b_i.shape[0])
                e = e/np.linalg.norm(e)*np.linalg.norm(b_i)*sigma
                delta = np.linalg.norm(e)
                b_m = b_i + e
                saveA[i] = A_n
                B[i] = b_m
                saveb_true[:, i] = sn
                saveb[:, i] = b_m
                savee[:, i] = e
                savedelta[i] = delta
                astra.projector.delete(proj_id)
                astra.matrix.delete(mat_id)
            A = sps.block_diag((saveA))    
            b = saveb.flatten(order ='F') 
            # xf = savex_true.flatten(order = 'F')
            truth = savex_true.reshape((nx, ny, nt), order='F').transpose((2,0,1))
        elif source=='real':
            import h5py
            nx, ny, nt =  140, 140, 8
            N = 140
            N_det = 140
            N_theta = 45
            theta = np.linspace(0,360,N_theta,endpoint=False)
            # Load measurement data as sinogram
            # data = spio.loadmat('./data/'+data_file) # scipy.io does not support Matlab struct
            data = h5py.File('./data/'+data_file,'r')
            CtData = data["CtData"]
            m = np.array(CtData["sinogram"]).T # strange: why is it transposed?
            # Load parameters
            param = CtData["parameters"]
            f = h5py.File('A_seqData.mat')
            fA = f["A"]
            # Extract information
            Adata = np.array(fA["data"])
            Arowind = np.array(fA["ir"])
            Acolind = np.array(fA["jc"])
            # Need to know matrix size (shape) somehow
            n_rows = N_det*N_theta # 6300
            n_cols = N*N
            Aloaded = sps.csc_matrix((Adata, Arowind, Acolind), shape=(n_rows, n_cols))
            saveA = list(range(nt))
            saveb = np.zeros((n_rows, nt))
            savee = np.zeros((n_rows, nt))
            savedelta = np.zeros((nt, 1))
            B = list(range(nt))
            for i in range(nt):
                tmp = m[45*(i):45*(i+1), :]
                b_i = tmp.flatten()
                sigma = 0.01 # noise level
                e = np.random.normal(0, 1, b_i.shape[0])
                e = e/np.linalg.norm(e)*np.linalg.norm(b_i)*sigma
                delta = np.linalg.norm(e)
                b_m = b_i + e
                saveA[i] = Aloaded
                B[i] = b_m
                saveb[:, i] = b_m
                savee[:, i] = e
                savedelta[i] = delta
            A = sps.block_diag((saveA))    
            b = saveb.flatten(order ='F')
            truth = None
        return A, b, saveA, B, nx, ny, nt, savedelta, truth
    
    def observe(self):
        """
        Observe image projections
        """
        A, b, AA, B, nx, ny, nt, delta, truth = self._gen_stempo()
        sz_x = (nx,ny) # I = np.prod(sz_x) = nx*ny
        sz_t = nt # J = sz_t
        ops_proj = AA; obs_proj = B
        # nzcov = np.cov(np.stack(obs_proj), rowvar=False)
        nzvar = np.var(np.stack(obs_proj), axis=0)
        return (ops_proj,obs_proj), nzvar, sz_x, sz_t, truth
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='stempo_'+self.data_src+'_obs'
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzvar=loaded['nzvar']; sz_x=loaded['sz_x']; sz_t=loaded['sz_t']; truth=loaded['truth']
            print('Observation file '+obs_file_name+' has been read!')
        except Exception as e:
            print(e); pass
            obs, nzvar, sz_x, sz_t, truth = self.observe()
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzvar=nzvar, sz_x=sz_x, sz_t=sz_t, truth=truth)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
    
        return obs, nzvar, sz_x, sz_t, truth
    
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
        try:
            from package.operators import time_derivative_operator
            L = time_derivative_operator(nx=sz_x[0], ny=sz_x[1], nt=sz_t)
        except Exception as e:
            print(e)
            _op = lambda n: sps.eye(n)-sps.eye(m=n,n=n,k=-1)
            D_x = sps.vstack((sps.kron(sps.eye(sz_x[0]),_op(sz_x[0])),sps.kron(_op(sz_x[1]),sps.eye(sz_x[0]))))
            D_t = _op(sz_t)[:-1]
            L = sps.vstack((sps.kron(sps.eye(sz_t),D_x),sps.kron(D_t,sps.eye(np.prod(sz_x)))))
        return L
    
    def reconstruct_anisoTV(self, iter=3):
        """
        Reconstruct images by anisoTV
        """
        A, b, AA, B, nx, ny, nt, delta, truth = self._gen_stempo()
        L = self._anisoTV((nx,ny),nt)
        try:
            from package.gks import GKS
            (xx, x_history, lambdah, lambda_history) = GKS(A, b, L, 1, iter)
        except Exception as e:
            print(e)
            from gks_tools import GKS
            xx = GKS(A, b, L, 1, iter)
        xx = np.reshape(xx, (nx, ny, nt), order="F")
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
            if save_imgs:  plt.savefig(save_path+'/stempo_'+str(i).zfill(len(str(rcstr_imgs.shape[2])))+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()

if __name__ == '__main__':
    np.random.seed(2022)
    import time
    from prior import *
    
    # define the misfit
    data_src = 'simulation'
    msft = misfit(data_src=data_src)
    # define the prior
    pri = prior(sz_x=msft.sz_x,sz_t=msft.sz_t)
    
    t0=time.time()
    # generate sample
    # u = np.random.rand(np.prod(msft.sz_x),msft.sz_t)
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
    
    # reconstruct the images by anisoTV
    xx=msft.reconstruct_anisoTV()
    # plot
    # import matplotlib.pyplot as plt
    msft.plot_reconstruction(xx, save_imgs=True, save_path='./reconstruction/anisoTV_'+msft.data_src)
    
    # evaluate the likelihood at anisoTV reconstruction
    u=xx.reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    nll=msft.cost(u)
    grad=msft.grad(u)
    print('The negative logarithm of likelihood at anisoTV reconstruction is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))
    
    # reconstruct the images by LSE
    x_hat=msft.reconstruct_LSE(lmda=10)
    # plot
    # import matplotlib.pyplot as plt
    msft.plot_reconstruction(x_hat, save_imgs=True, save_path='./reconstruction/LSE_'+msft.data_src)
    
    # evaluate the likelihood at anisoTV reconstruction
    u=x_hat.reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    nll=msft.cost(u)
    grad=msft.grad(u)
    print('The negative logarithm of likelihood at LSE reconstruction is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))