#!/usr/bin/env python
"""
Class definition of data-misfit for the dynamic non-linear example Navier-Stokes equation (NSE)
-----------------------------------------------------------------------------------------------
Created March 3, 2023 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import os
import numpy as np
import scipy as sp
import torch
from utilities3 import *
from torch.autograd.functional import jacobian, hvp, vhp

# self defined modules
from emulator import *

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
        self.trained_model = kwargs.pop('trained_model',None) # trained model
        self.data_set = kwargs.pop('data_set','V1e-4') # data set
        self.data_thinning = kwargs.pop('data_thinning',4) # data thinning
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.obs, nzcov, self.sz_x, self.sz_t, self.truth = self.get_obs(**kwargs)
        self.grid = self._get_grid()
        self.nzlvl = kwargs.pop('nzlvl',1.) # noise level
        # self.nzcov = self.nzlvl * max(torch.diag(nzcov)) * (nzcov + self.jit * torch.eye(nzcov.shape[0]))
        self.nzcov = self.nzlvl * max(torch.diag(nzcov)) * torch.eye(nzcov.shape[0],device=self.device)#.to_sparse_csr()
    
    def _load_model(self):
        """
        Load trained emulation model
        """
        # load model
        mdl_name='ns_fourier_'+self.data_set+{'V1e-3':'_T50_N4800_ep500_m12_w32','V1e-4':'_T20_N9800_ep200_m12_w32'}[self.data_set]
        if not os.path.exists('./model'): os.makedirs('./model')
        if not os.path.exists('./model/'+mdl_name):
            url='https://drive.google.com/drive/folders/1swLA6yKR1f3PKdYSKhLqK4zfNjS9pt_U'
            # import requests
            # r = requests.get(url+'/'+dat_name)
            # with open('./model/'+mdl_name+'.mat', "wb") as file:
            #     file.write(r.content)
            import gdown
            gdown.download_folder(url, quiet=True, use_cookies=False)
            print("NSE trained model downloaded.")
        model = torch.load('model/'+mdl_name, map_location=self.device)
        return model
    
    def _gen_solution(self):
        """
        Generate (emulated) NSE solution
        """
        # load data
        dat_name='ns_data_'+self.data_set+{'V1e-3':'_N5000_T50','V1e-4':'_N20_T50_R256test','V1e-5':'_N1200_T20'}[self.data_set]
        if not os.path.exists('./data'): os.makedirs('./data')
        if not os.path.exists('./data/'+dat_name+'.mat'):
            dat_name='NavierStokes_'+self.data_set+{'V1e-3':'_N5000_T50','V1e-4':'_N20_T50_R256_test','V1e-5':'_N1200_T20'}[self.data_set]
            print("downloading...")
            url='https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-'
            # import requests
            # r = requests.get(url+'/'+dat_name+'.zip')
            # with open('./data/'+dat_name+'.zip', "wb") as file:
            #     file.write(r.content)
            import gdown
            gdown.download_folder(url, quiet=True, use_cookies=False)
            print("NSE data downloaded.")
        reader = MatReader('./data/'+dat_name+'.mat')
        U = reader.read_field('u')[[0]]
        sub_s = self.data_thinning
        sub_t = self.data_thinning
        S = int(U.shape[1]/sub_s)
        T = 20
        T_in = 10
        indent = 1+int(np.log2(sub_t))
        truth = U[:,::sub_s,::sub_s, 3:T_in*4:4] #([0, T_in])
        obs = U[:,::sub_s,::sub_s, indent+T_in*4:indent+(T+T_in)*4:sub_t] #([T_in, T_in + T])
        # pad the location information (s,t)
        T = T * (4//sub_t)
        # test_a = truth.reshape(1,S,S,1,T_in).repeat([1,1,1,T,1])
        # gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        # gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
        # gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        # gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
        # gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
        # gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
        # test_a = torch.cat((gridx, gridy, gridt, test_a), dim=-1)
        # out = model(test_a)
        nzcov = torch.cov(obs.reshape(-1,T))
        return obs, nzcov, (S,S), T_in, truth.squeeze()
    
    def _get_grid(self):
        """
        Get grid data
        """
        S = self.sz_x; T = self.obs.shape[-1]
        gridx = torch.tensor(np.linspace(0, 1, S[0]), dtype=torch.float, device=self.device)
        gridx = gridx.reshape(1, S[0], 1, 1, 1).repeat([1, 1, S[1], T, 1])
        gridy = torch.tensor(np.linspace(0, 1, S[1]), dtype=torch.float, device=self.device)
        gridy = gridy.reshape(1, 1, S[1], 1, 1).repeat([1, S[0], 1, T, 1])
        gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float, device=self.device)
        gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S[0], S[1], 1, 1])
        return gridx, gridy, gridt
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='NSE_obs_'+self.data_set+('_thinning'+str(self.data_thinning) if self.data_thinning>1 else '')
        try:
            # loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.mat'),allow_pickle=True)
            # obs=loaded['obs']; nzcov=loaded['nzcov']; sz_x=loaded['sz_x']; sz_t=loaded['sz_t'];
            # grid=loaded['grid'];  truth=loaded['truth']
            reader = MatReader(os.path.join(obs_file_loc,obs_file_name+'.mat'),to_cuda=self.device=='cuda')
            obs = reader.read_field('obs'); nzcov=reader.read_field('nzcov'); truth=reader.read_field('truth')
            sz_x = reader.data['sz_x'][0]; sz_t = reader.data['sz_t'][0,0]
            print('Observation file '+obs_file_name+' has been read!')
        except Exception as e:
            print(e); pass
            obs, nzcov, sz_x, sz_t, truth = self._gen_solution()
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                # np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzcov=nzcov, sz_x=sz_x, sz_t=sz_t, grid=grid, truth=truth)
                sp.io.savemat(os.path.join(obs_file_loc,obs_file_name+'.mat'),mdict={'obs':obs.cpu().numpy(),
                                                                                     'nzcov':nzcov.cpu().numpy(),
                                                                                     'sz_x':sz_x,'sz_t':sz_t,
                                                                                     'truth':truth.cpu().numpy()})
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzcov, sz_x, sz_t, truth
    
    def _fwd_map(self, u):
        """
        Evaluate the emulated forward mapping
        """
        u_ = u if torch.is_tensor(u) else torch.tensor(u,dtype=torch.float,device=self.device)
        u_ = u_.reshape(1,self.sz_x[0],self.sz_x[1],1,self.sz_t).repeat([1,1,1,self.obs.shape[-1],1])
        u_ = torch.cat(self.grid+(u_,),dim=-1)
        return self.model(u_)
    
    def cost(self, u=None, obs=None):
        """
        Evaluate misfit function for given vector u.
        """
        if obs is None:
            if u.shape[0]!=np.prod(self.sz_x):
                u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
            obs = self._fwd_map(u)
        dif_obs = (obs - self.obs).reshape((-1,self.obs.shape[-1]))
        val = .5*torch.sum(dif_obs*torch.linalg.solve(self.nzcov,dif_obs))
        return val
    
    # def grad(self, u=None, obs=None):
    #     """
    #     Compute the gradient of misfit
    #     """
    #     if u.shape[0]!=np.prod(self.sz_x):
    #         u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
    #     u_ = torch.tensor(u,dtype=torch.float,requires_grad=True,device=self.device)
    #     val = self.cost(u_)
    #     val.backward()
    #     g = u_.grad
    #     return g # (I,J)
    
    def grad(self, u=None, obs=None):
        """
        Compute the gradient of misfit
        """
        if u.shape[0]!=np.prod(self.sz_x):
            u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
        u_ = torch.tensor(u,dtype=torch.float,requires_grad=True,device=self.device)
        g = jacobian(self.cost, u_)
        return g
    
    # def Hess(self, u=None, obs=None):
    #     """
    #     Compute the Hessian action of misfit
    #     """
    #     if u.shape[0]!=np.prod(self.sz_x):
    #         u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
    #     u_ = torch.tensor(u,dtype=torch.float,requires_grad=True,device=self.device)
    #     if obs is None:
    #         obs = self._fwd_map(u_)
    #     dif_obs = (obs - self.obs).reshape((-1,self.obs.shape[-1]))
    #     val = .5*torch.sum(dif_obs*torch.linalg.solve(self.nzcov,dif_obs))
    #     val.backward()
    #     g = u_.grad
    #     def hess(v):
    #         g.requires_grad = True
    #         if v.shape[:2]!=(np.prod(self.sz_x),self.sz_t):
    #             v=v.reshape((np.prod(self.sz_x),self.sz_t,-1),order='F') # (I,J,K)
    #         if v.ndim==2: v=v[:,:,None]
    #         v_ = torch.tensor(v,dtype=torch.float,requires_grad=False,device=self.device)
    #         gv = torch.sum(g[:,:,None]*v_,dim=(0,1))
    #         gv.backward()
    #         Hv = u_.grad
    #         return Hv
    #     return hess
    
    def Hess(self, u=None, obs=None):
        """
        Compute the Hessian action of misfit
        """
        if u.shape[0]!=np.prod(self.sz_x):
            u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
        u_ = torch.tensor(u,dtype=torch.float,requires_grad=True,device=self.device)
        def hess(v):
            if v.shape[:2]!=(np.prod(self.sz_x),self.sz_t):
                v=v.reshape((np.prod(self.sz_x),self.sz_t,-1),order='F') # (I,J,K)
            # if v.ndim==2: v=v[:,:,None]
            v_ = torch.tensor(v,dtype=torch.float,requires_grad=False,device=self.device)
            # import time
            # start = time.time()
            # Hv = hvp(self.cost, u_, v_)[1]
            # end = time.time()
            # print('Time: %.4f' % (end-start))
            # start = time.time()
            Hv = vhp(self.cost, u_ if v_.ndim==2 else u_[:,:,None].repeat([1,1,v_.shape[-1]]), v_)[1]
            # Hv = torch.stack([vhp(self.cost, u_ , v_[:,:,i])[1] for i in range(v_.shape[-1])],dim=2)
            # end = time.time()
            # print('Time: %.4f' % (end-start))
            return Hv.squeeze()
        return hess
    
    def plot_data(self, dat_imgs, save_imgs=False, save_path='./data'):
        """
        Plot the data.
        """
        if np.ndim(dat_imgs)!=3: dat_imgs=dat_imgs.reshape(np.append(self.sz_x,-1),order='F')
        # plot
        import matplotlib.pyplot as plt
        plt.set_cmap('Greys')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        for i in range(dat_imgs.shape[2]):
            plt.imshow(dat_imgs[:,:,i], origin='lower')
            plt.title('t = '+str(i),fontsize=16)
            if save_imgs: plt.savefig(save_path+'/NSE_'+str(i).zfill(len(str(dat_imgs.shape[2])))+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()
    
if __name__ == '__main__':
    np.random.seed(2022)
    import time
    from prior import *
    
    # define the misfit
    data_set='V1e-4'
    msft = misfit(data_set=data_set)
    # define the prior
    pri = prior(sz_x=msft.sz_x,sz_t=msft.sz_t)
    
    t0=time.time()
    # generate sample
    # u=np.random.rand(np.prod(msft.sz_x),msft.sz_t)
    u=pri.sample('fun').reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    # u=msft.truth.cpu().numpy().reshape(-1,msft.sz_t)
    nll=msft.cost(u).detach().cpu().numpy()
    grad=msft.grad(u).cpu().numpy()
    print('The negative logarithm of likelihood at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))
    hess=msft.Hess(u)
    # test
    # v=np.random.rand(np.prod(msft.sz_x),msft.sz_t)
    v=pri.sample('fun').reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    h=1e-5
    gradv_fd=(msft.cost(u+h*v).detach().cpu().numpy()-nll)/h
    gradv=np.sum(grad*v)
    rdiff_gradv=np.abs(gradv_fd-gradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    hessv_fd=(msft.grad(u+h*v).cpu().numpy()-grad)/h
    hessv=hess(v).cpu().numpy()
    rdiff_hessv=np.linalg.norm(hessv_fd-hessv)/np.linalg.norm(v)
    print('Relative difference of Hessian-action in a direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
    t1=time.time()
    print('time: %.5f'% (t1-t0))
    
    # plot data
    # msft.plot_data(dat_imgs=msft.truth.cpu().numpy(), save_imgs=True, save_path='./data/truth')
    # msft.plot_data(dat_imgs=msft.obs.cpu().numpy(), save_imgs=True, save_path='./data/obs')
    # msft.plot_data(dat_imgs=msft._fwd_map(msft.truth).detach().numpy(), save_imgs=True, save_path='./data/truth_fwd')