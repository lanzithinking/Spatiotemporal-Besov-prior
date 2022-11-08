#!/usr/bin/env python
"""
Geometric Infinite dimensional MCMC samplers
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Created March 12, 2016
-------------------------------
Modified Dec. 8, 2021 @ ASU
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "1.3"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; lanzithinking@outlook.com; slan@asu.edu"

import numpy as np
import timeit,time

def wMALA(u0,l0,grad,loglik,T,h=0.1):
    """
    Whitened Metropolis Adjusted Langevin Algorithm (wMALA)
    ---------------------------------------------------------
    inputs:
      u0: initial state of the parameters
      l0: initial log-likelihood
      grad: D phi(T(eps)) = d_phi/d_eps
      loglik: log-likelihood function
      T: map transforming white noise to (non)-Gaussian prior mu0
      h: step size
    outputs:
      u: new state of the parameter following lik*mu0
      l: new log-likelihood
      acpt: indicator of acceptance
    '''
    """
    beta = 4*np.sqrt(h)/(4+h)
    # sample velocity
    v=np.random.randn(*u0.shape)
    
    # generate proposal according to MALA scheme, modify T'(u), gradient to u
    
    u = (1-beta**2)**.5*u0 + beta*(v - np.sqrt(h)/2*grad(u0))
    
    # current energy I(u0, u)
    E_cur = -l0 + h/8*grad(u0).dot(grad(u0)) - np.sqrt(h)/2*grad(u0).dot((u - np.sqrt(1-beta**2)*u0)/beta) 

    # update loglik
    l=loglik(T(u))
    
    # updated energy I(u, u0)
    E_up = -l + h/8*grad(u).dot(grad(u)) - np.sqrt(h)/2*grad(u).dot((u0 - np.sqrt(1-beta**2)*u)/beta) 

    
    
    # Metropolis test
    logr = E_cur - E_up
    
    if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
        # accept
        acpt=True
    else:
        # reject
        u=u0; l=l0
        acpt=False
        
    # return 
    return u,l,acpt