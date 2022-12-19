#!/usr/bin/env python
"""
Dimension-Robust MCMC samplers
Victor Chen, Matthew M. Dunlop, Omiros Papaspiliopoulos, Andrew M. Stuart
--------------------------------
https://arxiv.org/abs/1803.03344
--------------------------------
"""
__author__ = "Shuyi Li"
__copyright__ = "Copyright 2022, The STBP Project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import timeit,time

def winfMALA(u0,l0,g0,loglik,T,h=0.1):
    """
    Whitened Infinite-dimensional Metropolis Adjusted Langevin Algorithm (wMALA)
    ---------------------------------------------------------
    inputs:
      u0: initial state of the parameters
      l0: initial log-likelihood
      g0: initial gradient of log-likelihood
      loglik: function for log-likelihood and its gradient
      T: map transforming white noise to (non)-Gaussian prior mu0 and its Jacobian
      h: step size
    outputs:
      u: new state of the parameter following lik*mu0
      l: new log-likelihood
      g: new gradient of log-likelihood
      acpt: indicator of acceptance
    '''
    """
    # fix weight parameter
    rth=np.sqrt(h)
    beta=4*rth/(4+h) # (0,1]
    
    # sample velocity
    v=np.random.randn(*u0.shape)
    
    # update velocity
    v+=rth/2*g0
    
    # current energy
    I_cur = -l0 + g0.dot(-rth/2*v+h/8*g0)
    
    # generate proposal according to MALA scheme, modify T'(u), gradient to u
    u = np.sqrt(1-beta**2)*u0 + beta*v
    
    # update velocity
    v = -np.sqrt(1-beta**2)*v + beta*u0
    
    # update geometry
    l,g=loglik(u,T,grad=True)
    
    # new energy
    I_prp = -l + g.dot(-rth/2*v+h/8*g)
    
    # Metropolis test
    logr=-I_prp+I_cur
    
    if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
        # accept
        acpt=True
    else:
        # reject
        u=u0; l=l0; g=g0
        acpt=False
        
    # return 
    return u,l,g,acpt