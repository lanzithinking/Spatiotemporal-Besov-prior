import numpy as np
import scipy as sp
import scipy.io as spio
from scipy.fftpack import dct, idct
import matplotlib
import matplotlib.pyplot as plt
import astra
from scipy.sparse import csr_matrix
import requests
import scipy as sp
from scipy import sparse
import numpy as np
import os, sys
sys.path.insert(0,'/Users/mirjetapasha/Documents/Research_Projects/BesovPrior_September_9/Spatiotemporal-Besov-prior/STEMPO/')
from os.path import exists
import scipy.sparse as sps
# import scipy.linalg as spla
import scipy.io as spio
import numpy as np
import h5py
import os, sys
from os.path import exists
import scipy.sparse as sps
import pylops
from gks_l import *
# functions to generate emoji data are stored in io_l.py
from io11 import *
from operators_l import *
from hlp117 import *
from scipy.sparse import coo_matrix, block_diag
# from scipy.linalg import block_diag
# self defined modules
# from gks_tools import *
# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class misfit_STEMPO(object):
    
    def __init__(self, **kwargs):
        """
        Initialize data-misfit class with information of observations.
        """
        self.nzlvl = kwargs.pop('nzlvl',1.) # noise level
        # self.jit = kwargs.pop('jit',1e-3) # jitter to the noise covariance
        # get observations
        self.obs, nzcov, self.sz_x, self.sz_t = self.get_obs(**kwargs)
        # self.nzcov = self.nzlvl * max(np.diag(nzcov)) * (nzcov + self.jit * sps.eye(nzcov.shape[0]))
#         self.nzcov = self.nzlvl * max(np.diag(nzcov)) * ( sps.eye(nzcov.shape[0]))
    
    def _gen_stempo(self):
        """
        Generate stempo observations
        """
#         if not os.path.exists('./stempo'): os.makedirs('./stempo')
#         if not os.path.exists('./stempo/stempo_ground_truth_2d_b4.mat'):
#             import requests
#             print("downloading...")
#             r = requests.get('https://zenodo.org/record/7147139#.Y0Bba-zMLeo/stempo_ground_truth_2d_b4.mat')
#             with open('.stempo/stempo_ground_truth_2d_b4.mat', "wb") as file:
#                 file.write(r.content)
#             print("Stempo data downloaded.")
        truth = spio.loadmat('stempo_ground_truth_2d_b4.mat')
        image = truth['obj']
        nx, ny, nt = 560, 560, 10;
        angleVector = list(range(nt))
        for t in range(nt):
            angleVector[t] = np.linspace(t, 360 - 4*nt + 4*t, num = nt, endpoint = True)
        angleVectorRad = np.deg2rad(angleVector)
                # Generate matrix versions of the operators and a large bidiagonal sparse matrix
        N = nx         # object size N-by-N pixels
        p = int(np.sqrt(2)*N)    # number of detector pixels
        # view angles
        theta = angleVectorRad#[0]#np.linspace(0, 2*np.pi, q, endpoint=False)   # in rad
        q = theta.shape[0]          # number of projection angles
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
        for i in range(nt):
            proj_geom = astra.create_proj_geom('fanflat', detector_pixel_size, p, theta[i], source_origin, detector_origin)
            vol_geom = astra.create_vol_geom(N, N)
            proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
            mat_id = astra.projector.matrix(proj_id)
            A_n = astra.matrix.get(mat_id)
            x_true = image[:, :, i]
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
        A = block_diag((saveA))    
        b = saveb.flatten(order ='F') 
        xf = savex_true.flatten(order = 'F')
        return A, b, saveA, B, nx, ny, nt, savedelta, saveb
    
    def observe(self):
        """
        Observe image projections
        """
        A, b, AA, B, nx, ny, nt, delta, saveB = self._gen_stempo()
        print('In Observe')
#         print(B[0].shape)
        sz_x = (nx,ny) # I = np.prod(sz_x) = nx*ny
        sz_t = nt # J = sz_t
        ops_proj = AA; obs_proj = B
        # nzcov = np.cov(np.stack(obs_proj), rowvar=True)
        tmp = ops_proj[0].shape[0]
        nzcov = np.identity(tmp)
        return (ops_proj,obs_proj), nzcov, sz_x, sz_t
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='stempo_obs'
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
        print('Computing inverse variance')
        a = np.stack(dif_obs).shape[0]
        nzcov = np.identity(a)
        val = .5*np.sum(dif_obs*spsla.spsolve(nzcov,dif_obs))
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
        a = np.stack(dif_obs).shape[0]
        nzcov = np.identity(a)
        for i in range(self.sz_t):
            # dif_obs = ops_proj[i].dot(u[:,i]) - obs_proj[i]
            # g.append( ops_proj[i].T.dot(spla.solve(self.nzcov,dif_obs,sym_pos=True)) )
            g.append( ops_proj[i].T.dot(spsla.spsolve(nzcov,dif_obs[:,i])) )
        return np.stack(g).T # (I,J)
    
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
        A, b, AA, B, nx, ny, nt, delta, saveB = self._gen_stempo()
        L = time_derivative_operator(nx, ny, nt)
        # L = pylops.Identity(nx*nx, dtype='float32')
        (xx, x_history, lambdah, lambda_history) = GKS(A, b, L, 1, 3) 
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
    msft = misfit_STEMPO()
    # define the prior
     
    # pri = prior(sz_x=msft.sz_x,sz_t=msft.sz_t)
    A, b, saveA, B, nx, ny, nt, savedelta, saveB = msft._gen_stempo()
    u = np.random.rand(np.prod(msft.sz_x),msft.sz_t)
    cst = msft.cost(u)
    grd = msft.grad(u)