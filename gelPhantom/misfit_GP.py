from cil.framework import AcquisitionGeometry
from cil.optimisation.algorithms import PDHG
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.optimisation.functions import IndicatorBox, BlockFunction, L2NormSquared, MixedL21Norm
from cil.io import NEXUSDataWriter, NEXUSDataReader
from cil.processors import Slicer
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.plugins.ccpi_regularisation.functions import FGP_dTV, FGP_TV
from cil.utilities.display import show2D, show_geometry
from cil.utilities.jupyter import islicer
from utilities_dynamic_ct import read_frames, read_extra_frames
from IPython.display import clear_output
import wget
import numpy as np
import scipy as sp
import scipy.sparse as sps
# import scipy.linalg as spla
import scipy.io as spio
import scipy.sparse.linalg as spsla
# import h5py # needed to unpack downloaded gelPhantom data (in .mat format)
import os, sys
sys.path.insert(0, "/home/mpasha3/CIL/CILDemos/CIL-Demos/examples/3_Multichannel")
## Pylops and psandqs imports
from psandqs.gks import GKS
from psandqs.decompositions import generalized_golub_kahan
import pylops
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter('once')
class misfit_GP(object):
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
        self.obs, nzcov, self.sz_x, self.sz_t, self.ig, self.ag, self.ig_small, self.ag_small, self.A_proj, self.data = self.get_obs(**kwargs)
        # self.nzcov = self.nzlvl * max(np.diag(nzcov)) * (nzcov + self.jit * sps.eye(nzcov.shape[0]))
        self.nzcov = self.nzlvl * max(np.diag(nzcov)) * ( sps.eye(nzcov.shape[0]))
    
    def _gen_gelPhantom(self):
        """
        Generate gelPhantom observations
        """
        # MP: Download the data from https://zenodo.org/record/3696817/files/GelPhantomData_b4.mat
        path = os.path.abspath("/home/mpasha3/CIL/CILDemos/CIL-Demos/examples/3_Multichannel/data")
        data_mat = "GelPhantomData_b4"
        file_info = read_frames(path, data_mat)
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
        B = np.zeros((data.shape[1]*data.shape[2], frames))
        for i in range(frames):
            temp = ((data.array)[i, :, :]).flatten()
            B[:, i] = temp
        return A, data, AA, B, n_x, n_y, n_t, ig, ag, ig_small, ag_small
    
    def observe(self):
        """
        Observe image projections
        """
        A, b, AA, B, nx, ny, nt, geometry_x, geometry_b, geometry_x_small, geometry_b_small = self._gen_gelPhantom()
        sz_x = (nx,ny) # I = np.prod(sz_x) = nx*ny
        sz_t = nt # J = sz_t
        ops_proj = AA; obs_proj = B
        nzcov = np.cov(np.stack(obs_proj), rowvar=True)
        
        A_cil_forward = lambda x: self.ten_to_vec( A.direct( self.vec_to_ten(x, geometry_x)[0] ), geometry_b)[0]
        A_cil_backward = lambda b: self.ten_to_vec( A.adjoint( self.vec_to_ten(b, geometry_b)[0] ), geometry_x)[0]
        A_projection_operator = pylops.FunctionOperator(A_cil_forward, A_cil_backward, np.prod(geometry_b.shape), np.prod(geometry_x.shape) )
        
#         A_cil_forward_small = lambda x: mat_to_vec(AA{1}.direct( vec_to_mat(x, geometry_x_small)[0] ), geometry_b_small)[0]
#         A_cil_backward_small = lambda b: mat_to_vec( AA{1}.adjoint( vec_to_mat(b, geometry_b_small)[0] ), geometry_x_small)[0]
#         A_projection_operator_small = pylops.FunctionOperator(cil_forward_small, cil_backward_small, np.prod(geometry_b_small.shape), np.prod(geometry_x_small.shape) )
        
        return (ops_proj,obs_proj), nzcov, sz_x, sz_t, geometry_x, geometry_b, geometry_x_small, geometry_b_small, A_projection_operator, b
    
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs, nzcov, sz_x, sz_t, ig, ag, ig_small, ag_small, A_proj, b = self.observe()
        return obs, nzcov, sz_x, sz_t, ig, ag, ig_small, ag_small, A_proj, b
    
    def cost(self, u):
        """
        Evaluate misfit function for given images (vector) u.
        """
        if len(u.shape) < 2:
            u = u.reshape((np.prod(self.sz_x)), self.sz_t)
        ops_proj, obs_proj = self.obs
        for i in range(self.sz_t):
            aaa = self.mat_to_vec(ops_proj[i].direct(self.vec_to_mat(u[:, i], self.ig_small)[0]), self.ag_small)[0]
            obs = np.zeros(((aaa.shape)[0],self.sz_t))
            obs[:, i] = aaa
        dif_obs = obs - obs_proj 
        covMat = self.nzcov
        sparse_inv_cov = np.diag(1/covMat.diagonal())
        temp = sparse_inv_cov@dif_obs
        aa = dif_obs*temp
        val = .5*np.sum(aa)
        return val
    
    def grad(self, u):
            """
            Compute the gradient of misfit
            """
            if len(u.shape) < 2:
                u = u.reshape((np.prod(self.sz_x)), self.sz_t)
            for i in range(self.sz_t):
                aaa = self.mat_to_vec(ops_proj[i].direct(self.vec_to_mat(u[:, i], self.ig_small)[0]), self.ag_small)[0]
                obs = np.zeros(((aaa.shape)[0], self.sz_t))
                obs[:, i] = aaa
            dif_obs = obs - obs_proj 
            sparse_inv_cov = np.diag(1/np.diag(nzcov))
            temp = sparse_inv_cov@dif_obs
                # g.append( ops_proj[i].T.dot(spla.solve(self.nzcov,dif_obs,sym_pos=True)))
            g1 = []
            for i in range(sz_t):
                temp1 = self.mat_to_vec(ops_proj[i].adjoint(self.vec_to_mat(temp[:, i], self.ag_small)[0]), self.ig_small)[0]
                print(temp1.shape)
                g1.append(temp1)
            return np.stack(g1).T # (I,J)
    
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
    
    def reconstruct_anisoTV(self):
        """
        Reconstruct images by anisoTV
        """
        (ops_proj,obs_proj), nzcov, sz_x, sz_t, geometry_x, geometry_b, geometry_x_small, geometry_b_small, A_projection_operator, b  =  self.get_obs()
        data_f = b.array.reshape(-1,1)
        L = self._anisoTV((nx,ny),nt)
        (x, x_history, lambdah, lambda_history) = GKS(A_projection_operator, data_f,  L, 1, 10)
        xx = np.reshape(x, (nx, ny, nt), order="F")
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
   
    def plot_reconstruction(self, rcstr_imgs, save_imgs = True, save_path='./reconstructionGelPhantom'):
            """
            Plot the reconstruction.
            """
            #     if np.ndim(rcstr_imgs)!=3: rcstr_imgs=rcstr_imgs.reshape(np.append(self.sz_x,self.sz_t),order='F')
            #     # plot
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            for i in range(rcstr_imgs.shape[2]):
                show2D(rcstr_imgs[:, :, i], num_cols=4, origin="upper",fix_range=(0,0.065),
                        cmap="inferno", title="Time-frame {}".format(i), size=(25, 20))
                if save_imgs: plt.savefig(save_path+'/gelPhantom_'+str(i).zfill(len(str(rcstr_imgs.shape[2])))+'.png',bbox_inches='tight')

msft = misfit_GP()
u=np.random.rand(np.prod(msft.sz_x),msft.sz_t)
A, b, AA, B, nx, ny, nt, ig, ag, ig_small, ag_small = msft._gen_gelPhantom()
(ops_proj,obs_proj), nzcov, sz_x, sz_t, geometry_x, geometry_b, geometry_x_small, geometry_b_small, A_proj_op, data = msft.get_obs() 
# x0 = ig.allocate
x = msft.reconstruct_anisoTV()
# data_f = B.reshape(-1,1)
costt = msft.cost(u)
msft.plot_reconstruction(x)
# dd = costt.diagonal()
# dd.shape