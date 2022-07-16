#!/usr/bin/env python
"""
Class definition of the dynamic linear inverse problem of emoji.
Shiwei Lan @ ASU 2022
--------------------------------------------------------------------------
Created July 5, 2022 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
from scipy import optimize
import os

# self defined modules
from prior import *
from misfit import *
# from posterior import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class emoji:
    def __init__(self,**kwargs):
        """
        Initialize the dynamic inverse problem by defining the prior model and the misfit (likelihood) model.
        """
        # define the inverse problem with prior, and misfit
        seed = kwargs.pop('seed',2022)
        self.setup(seed,**kwargs)
    
    def setup(self, seed=2022, **kwargs):
        """
        Set up the prior and the likelihood (misfit: -log(likelihood)) and posterior
        """
        # set (common) random seed
        np.random.seed(seed)
        sep = "\n"+"#"*80+"\n"
        # set misfit
        self.misfit = misfit(**kwargs)
        print('\nLikelihood model is obtained.')
        # set prior
        self.prior = prior(sz_x=self.misfit.sz_x,sz_t=self.misfit.sz_t,**kwargs)
        print('\nPrior model is specified.')
        # set low-rank approximate Gaussian posterior
        # self.post_Ga = Gaussian_apx_posterior(self.prior,eigs='hold')
        # print('\nApproximate posterior model is set.\n')
        if kwargs.pop('init_param',False):
            # obtain an initial parameter from a rough reconstruction
            self._init_param()
        # self.prior.mean = self.init_parameter
    
    def _init_param(self,opt='LSE'):
        """
        Initialize parameter with a quick but rough reconstruction
        """
        reconstruction_method='reconstruct_'+opt
        if hasattr(self.misfit, reconstruction_method):
            reconstruct=getattr(self.misfit,reconstruction_method)
        self.init_parameter = self.prior.fun2vec(reconstruct())
    
    def _get_misfit(self, parameter, MF_only=True):
        """
        Compute the misfit (default), or the negative log-posterior for given parameter.
        """
        # evaluate data-misfit function
        msft = self.misfit.cost(self.prior.vec2fun(parameter))
        if not MF_only: msft += self.prior.cost(parameter)
        return msft
    
    def _get_grad(self, parameter, MF_only=True):
        """
        Compute the gradient of misfit (default), or the gradient of negative log-posterior for given parameter.
        """
        # obtain the gradient
        grad = self.prior.fun2vec(self.misfit.grad(self.prior.vec2fun(parameter)).flatten(order='F'))
        if not MF_only: grad += self.prior.grad(parameter)
        return grad

    def _get_HessApply(self, parameter=None, MF_only=True):
        """
        Compute the Hessian apply (action) for given parameter,
        default to the Gauss-Newton approximation.
        """
        raise NotImplementedError('HessApply not implemented.')
    
    def get_geom(self,parameter=None,geom_ord=[0],**kwargs):
        """
        Get necessary geometric quantities including log-likelihood (0), adjusted gradient (1), 
        Hessian apply (1.5), and its eigen-decomposition using randomized algorithm (2).
        """
        if parameter is None:
            parameter=self.prior.mean
        loglik=None; agrad=None; HessApply=None; eigs=None;
        
        # # convert parameter vector to function
        # parameter = self.prior.vec2fun(parameter)
        
        # get log-likelihood
        if any(s>=0 for s in geom_ord):
            loglik = -self._get_misfit(parameter, **kwargs)
        
        # get gradient
        if any(s>=1 for s in geom_ord):
            agrad = -self._get_grad(parameter, **kwargs)
        
        # get Hessian Apply
        if any(s>=1.5 for s in geom_ord):
            pass
        
        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        if any(s>1 for s in geom_ord):
            pass
        
        return loglik,agrad,HessApply,eigs
    
    def get_eigs(self,parameter=None,**kwargs):
        """
        Get the eigen-decomposition of Hessian action directly using randomized algorithm.
        """
        raise NotImplementedError('eigs not implemented.')
    
    def get_MAP(self,SAVE=False,*kwargs):
        """
        Get the maximum a posterior (MAP).
        """
        import time
        sep = "\n"+"#"*80+"\n"
        print( sep, "Find the MAP point", sep)
        # set up initial point
        # param0 = self.prior.sample('vec')
        if not hasattr(self, 'init_parameter'): self._init_param()
        param0 = self.init_parameter #+ .1*self.prior.sample('vec',0)
        fun = lambda parameter: self._get_misfit(parameter, MF_only=False)
        grad = lambda parameter: self._get_grad(parameter, MF_only=False)
        global Nfeval
        Nfeval=1
        def call_back(Xi):
            global Nfeval
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], fun(Xi)))
            Nfeval += 1
        print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
        # solve for MAP
        start = time.time()
        # res = optimize.minimize(fun, param0, method='BFGS', jac=grad, callback=call_back, options={'maxiter':100,'disp':True})
        res = optimize.minimize(fun, param0, method='L-BFGS-B', jac=grad, callback=call_back, options={'maxiter':1000,'disp':True})
        # res = optimize.minimize(fun, param0, method='Newton-CG', jac=grad, callback=call_back, options={'maxiter':100,'disp':True})
        end = time.time()
        print('\nTime used is %.4f' % (end-start))
        # print out info
        if res.success:
            print('\nConverged in ', res.nit, ' iterations.')
        else:
            print('\nNot Converged.')
        print('Final function value: %.4f.\n' % res.fun)
        
        MAP = res.x
        
        if SAVE:
            import pickle
            fld_name='properties'
            self._check_folder(fld_name)
            f = open(os.path.join(fld_name,'MAP.pckl'),'wb')
            pickle.dump(MAP, f)
            f.close()
        
        return MAP
    
    def _check_folder(self,fld_name='result'):
        """
        Check the existence of folder for storing result and create one if not
        """
        if not hasattr(self, 'savepath'):
            cwd=os.getcwd()
            self.savepath=os.path.join(cwd,fld_name)
        if not os.path.exists(self.savepath):
            print('Save path does not exist; created one.')
            os.makedirs(self.savepath)
    
    def test(self,h=1e-4):
        """
        Demo to check results with the exact method against the finite difference method.
        """
        # random sample parameter
        # parameter = self.prior.sample('vec')
        if not hasattr(self, 'init_parameter'): self._init_param()
        parameter = self.init_parameter
        
        # MF_only = True
        import time
        # obtain the geometric quantities
        print('\n\nObtaining geometric quantities by direct calculation...')
        start = time.time()
        loglik,grad,_,_ = self.get_geom(parameter,geom_ord=[0,1])
        end = time.time()
        print('Time used is %.4f' % (end-start))
        
        # check with finite difference
        print('\n\nTesting against Finite Difference method...')
        start = time.time()
        # random direction
        v = self.prior.sample('vec')
        ## gradient
        print('\nChecking gradient:')
        parameter_p = parameter + h*v
        loglik_p = -self._get_misfit(parameter_p)
#         parameter_m = parameter - h*v
#         loglik_m = -self._get_misfit(parameter_m)
        dloglikv_fd = (loglik_p-loglik)/h
        dloglikv = grad.dot(v.flatten())
        rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/np.linalg.norm(v)
        print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
        end = time.time()
        print('Time used is %.4f' % (end-start))
    
if __name__ == '__main__':
    # set up random seed
    seed=2022
    np.random.seed(seed)
    # define Bayesian inverse problem
    spat_args={'basis_opt':'Fourier','l':1,'s':2,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    store_eig = True
    emj = emoji(spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)
    # test
    emj.test(1e-8)
    # obtain MAP
    map_v = emj.get_MAP(SAVE=True)
    print('MAP estimate: '+(min(len(map_v),10)*"%.4f ") % tuple(map_v[:min(len(map_v),10)]) )
    # #  compare it with the truth
    # true_param = emj.misfit.truth # no truth
    # map_f = emj.prior.vec2fun(map_v).reshape(true_param.shape)
    # relerr = np.linalg.norm(map_f-true_param)/np.linalg.norm(true_param)
    # print('Relative error of MAP compared with the truth %.2f%%' % (relerr*100))
    # # report the minimum cost
    # # min_cost = emj._get_misfit(map_v)
    # # print('Minimum cost: %.4f' % min_cost)
    # plot MAP
    map_f = emj.prior.vec2fun(map_v).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    emj.misfit.plot_reconstruction(rcstr_imgs=map_f, save_imgs=True, save_path='./reconstruction/MAP')