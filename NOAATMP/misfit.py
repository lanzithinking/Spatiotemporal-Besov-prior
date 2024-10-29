#!/usr/bin/env python
"""
Class definition of data-misfit for the example of NOAA temperature.
-------------------------------------------------------------------------
Created October 15, 2024 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import scipy.sparse as sps
# import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import os
import matplotlib.pyplot as plt

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
        self.hldt_pcnt = kwargs.pop('holdout_percent',0)
        self.obs, self.nzvar, self.sz_x, self.sz_t, self.missing_idx, self.holdout_msk = self.get_obs(**kwargs)
        if kwargs.pop('impute_missing',False):
            self.obs = self._impute(**kwargs)
            self.missing_idx = None
        self.times = np.linspace(0,1,self.sz_t,endpoint=False)
        self.times += self.times[1]
        self.nzlvl = kwargs.pop('nzlvl',1.) # noise level
        self.nzvar *= self.nzlvl
        # self.nzcov = max(self.nzvar) * ( sps.eye(nzvar.shape[0], format='csr'))
        self.nzcov = sps.spdiags(self.nzvar[None,:], 0, format='csr')
    
    def _gen_noaatmp(self):
        """
        Generate NOAA temperature observations
        """
        dat_file='NOAA_TMP.npz'
        dat = np.load('./data/'+dat_file, allow_pickle=True)
        temp, loc, time = dat['temp'], dat['loc'], dat['time']
        temp = temp.reshape((72,36,240), order='F')[36:,5:33]
        lon = np.linspace(182.5,357.5, num=36)
        lat = np.linspace(-62.5,72.5, num=28)
        return temp, lon, lat, time
    
    def observe(self):
        """
        Observe temperature and sample hold-out data
        """
        temp, lon, lat, time = self._gen_noaatmp()
        obs = temp.reshape((-1,temp.shape[-1]), order='F')
        nzvar = np.nanvar(obs, axis=-1)
        missing_idx = np.argwhere(np.isnan(obs))
        return obs, nzvar, lon, lat, missing_idx
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        os.makedirs(obs_file_loc, exist_ok=True)
        obs_file_name='noaatmp_obs'
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzvar=loaded['nzvar']; lon=loaded['lon']; lat=loaded['lat']; missing_idx=loaded['missing_idx']
            print('Observation file '+obs_file_name+' has been read!')
        except Exception as e:
            print(e); pass
            obs, nzvar, lon, lat, missing_idx = self.observe()
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzvar=nzvar, lon=lon, lat=lat, missing_idx=missing_idx)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        self.lon, self.lat = lon, lat
        sz_x = len(lon), len(lat) # I = np.prod(sz_x)
        sz_t = obs.shape[-1] # J = sz_t
        holdout_msk = np.full(obs.shape, False)
        if self.hldt_pcnt>0:
            holdout_idx = np.argwhere(~np.isnan(obs))
            holdout_idx = np.random.default_rng(kwargs.pop('random_seed',2024)).choice(holdout_idx, size=int(len(holdout_idx)*self.hldt_pcnt), replace=False)
            holdout_msk[holdout_idx[:,0],holdout_idx[:,1]] = True
        return obs, nzvar, sz_x, sz_t, missing_idx, holdout_msk
    
    def cost(self, u=None, obs=None):
        """
        Evaluate misfit function for given images (vector) u.
        """
        if obs is None:
            if u.shape[0]!=np.prod(self.sz_x):
                u=u.reshape((np.prod(self.sz_x),-1),order='F') # (I,J)
            obs = u
        dif_obs = obs - self.obs
        if self.hldt_pcnt>0: dif_obs[self.holdout_msk] = 0
        # val = .5*np.sum(dif_obs*spsla.spsolve(self.nzcov,dif_obs))
        val = 0.5*np.nansum(dif_obs**2/(self.nzvar if self.nzvar.size==1 else self.nzvar[:,None]))
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
        if self.hldt_pcnt>0: dif_obs[self.holdout_msk] = 0
        # g = spsla.spsolve(self.nzcov,dif_obs)
        g = dif_obs/(self.nzvar if self.nzvar.size==1 else self.nzvar[:,None])
        if self.missing_idx is not None: g[self.missing_idx[:,0],self.missing_idx[:,1]] = 0
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
            if self.missing_idx is not None: Hv[self.missing_idx[:,0],self.missing_idx[:,1]] = 0
            if self.hldt_pcnt>0: Hv[self.holdout_msk] = 0
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
    
    def _impute(self, dat=None, axis=0, method='mean', **kwargs):
        """
        Impute missing values.
        """
        if dat is None: dat = self.obs.copy()
        if axis==0 or (type(axis) is list and 0 in axis): # over locations
            for t in np.unique(self.missing_idx[:,1]):
                mis_id_t = self.missing_idx[self.missing_idx[:,1]==t,0]
                imputed_t = 0 if method=='zero' else getattr(np,'nan'+method)(self.obs[:, t])
                dat[mis_id_t, t] = imputed_t
        if axis==1 or (type(axis) is list and 1 in axis): # over time
            for s in np.unique(self.missing_idx[:,0]):
                mis_id_s = self.missing_idx[self.missing_idx[:,0]==s,1]
                imputed_s = 0 if method=='zero' else getattr(np,'nan'+method)(self.obs[s, :])
                dat[s, mis_id_s] = imputed_s
        return dat
    
    def reconstruct_LSE(self, lmda=0, impute=False, **kwargs):
        """
        Reconstruct images by least square estimate.
        """
        obs = self._impute(**kwargs) if impute else self.obs.copy()
        x_hat = []
        for j in range(self.sz_t):
            mis_id_j = self.missing_idx[self.missing_idx[:,1]==j,0]
            msk_j = np.ones(np.prod(self.sz_x))
            obs_j = obs[:,j]
            if not impute:
                msk_j[mis_id_j] = 0
                obs_j[mis_id_j] = 0 # mask missing values
            x_hat.append(spsla.lsqr(sps.spdiags(msk_j[None,:], 0),obs_j,damp=lmda)[0])
        return np.stack(x_hat).T
    
    def plot_reconstruction(self, rcstr_imgs, save_imgs=False, save_path='./reconstruction', **kwargs):
        """
        Plot the reconstruction.
        """
        if np.ndim(rcstr_imgs)!=3: rcstr_imgs=rcstr_imgs.reshape(np.append(self.sz_x,self.sz_t),order='F')
        snap_idx = kwargs.pop('snap_idx', np.arange(self.sz_t))
        time_lbls = self._gen_noaatmp()[-1]
        # plot
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        for t in snap_idx:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.plot_data(mapmat=rcstr_imgs[:,:,t], save_img=save_imgs, save_path=save_path, ax=ax, time_label=time_lbls[t,0], **kwargs)
            if kwargs.get('anim',True):
                plt.pause(.2)
                plt.draw()
    
    def plot_data(self, mapmat=None, save_img=False, save_path='./data', **kwargs):
        """
        Plot the data on map.
        """
        from matplotlib.colors import LinearSegmentedColormap
        # get data
        if mapmat is None:
            temp, lon, lat, time = self._gen_noaatmp()
            i_plot = 228
            print('Plot data on {}'.format(time[i_plot]))
            mapmat = temp[:,:,i_plot]
            fig, ax = plt.subplots(figsize=(10, 6))
            levels = np.linspace(-6, 6, 81)
        else:
            lon, lat = self.lon, self.lat
            ax = kwargs.pop('ax', plt.gca())
            levels = np.linspace(np.nanmin(mapmat), np.nanmax(mapmat), 81)  # Contour levels
        # Define the custom color palette (similar to colorRampPalette in R)
        colors = ['black', 'blue', 'darkgreen', 'green', 'yellow', 'pink', 'red', 'maroon']
        rgb_palette = LinearSegmentedColormap.from_list("custom_palette", colors)
        # set figure
        # Create the map with Basemap, Pacific-centered (lon_0=180)
        self._plot_map(ax=ax,
                       llcrnrlon=np.nanmin(lon), urcrnrlon=np.nanmax(lon),
                       llcrnrlat=np.nanmin(lat), urcrnrlat=np.nanmax(lat))
                       # suppress_ticks=False)
        # Create the filled contour plot
        contour = ax.contourf(lon, lat, mapmat.T, levels=levels, cmap=rgb_palette)
        # contour = ax.contourf(lon, lat, mapmat.T, vmin=-6, vmax=6, levels=81, cmap=rgb_palette)
        # Add colorbar with degree Celsius symbol (similar to key.title and key.axes in R)
        cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
        cbar.ax.set_title(u"°C", fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        if kwargs.pop('label', True):
            # Add labels and title
            plt.title("Temperature anomalies in "+kwargs.get('time_label','Jan 2018'), fontsize=16)
            plt.xlabel("Longitude", fontsize=14)
            plt.ylabel("Latitude", fontsize=14)
        # Customize axes tick sizes (like cex.axis and axis in R)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # Add gridlines
        plt.grid(True)
        if save_img:
            plt.savefig(os.path.join(save_path,'noaa_obs'+kwargs.get('time_label','')+'.png'),bbox_inches='tight')
        else:
            # Show the plot
            plt.show()
    
    def _plot_map(self, ax=None, **kwargs):
        from mpl_toolkits.basemap import Basemap
        if ax is None: ax=plt.gca()
        m = Basemap(projection='cyl', ax=ax, **kwargs)
        # Draw coastlines and countries
        m.drawcoastlines()
        m.drawcountries()
    
if __name__ == '__main__':
    np.random.seed(2022)
    import time
    from prior import *
    
    # define the misfit
    holdout_percent = 0.1
    msft = misfit(holdout_percent=holdout_percent)
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
    
    # plot the data on map
    # msft.plot_data(save_img=False, save_path='./data')
    
    # reconstruct the images by LSE
    x_hat=msft.reconstruct_LSE(lmda=1)
    # plot
    # msft.plot_reconstruction(msft.obs, save_imgs=True, save_path='./reconstruction/obs', snap_idx=np.linspace(0,msft.sz_t,num=10,endpoint=False).astype(int))
    # msft.plot_reconstruction(x_hat, save_imgs=True, save_path='./reconstruction/LSE', snap_idx=np.linspace(0,msft.sz_t,num=10,endpoint=False).astype(int))
    # reconstruct the images by LSE with imputation
    x_hat=msft.reconstruct_LSE(lmda=1, impute=True)
    msft.plot_reconstruction(x_hat, save_imgs=True, save_path='./reconstruction/LSE_impute', snap_idx=np.linspace(0,msft.sz_t,num=10,endpoint=False).astype(int))
    
    # evaluate the likelihood at reconstruction
    u=x_hat.reshape((np.prod(msft.sz_x),msft.sz_t),order='F')
    nll=msft.cost(u)
    grad=msft.grad(u)
    print('The negative logarithm of likelihood at LSE reconstruction is %0.4f, and the L2 norm of its gradient is %0.4f' %(nll,np.linalg.norm(grad)))