#!/usr/bin/env python
"""
function to compute the pdf and sample mvpe

---------------------------------------------------------------
Created June 15, 2022 for project of Bayesian Spatiotemporal Besov Prior 
"""
__author__ = "Shuyi Li"
__credits__ = 'https://github.com/ecbrown/LaplacesDemon/blob/master/R/distributions.R'
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"


import numpy as np
import math


def dmvpe(x=np.zeros((1,2)), mu=np.zeros((1,2)), Sigma=np.diag([1,1]), kappa=1, log=False):
    if ~isinstance(x, np.ndarray):
        x = np.array(x)
    if ~isinstance(mu, np.ndarray):
        mu = np.array(mu).reshape(x.shape[0], x.shape[1])
    if Sigma is None:
        Sigma = np.diag(np.ones(x.shape[1]))
    if ~isinstance(Sigma, np.ndarray):
        Sigma = np.array(Sigma)
    try:
        np.linalg.cholesky(Sigma)
    except Exception as err:
        #print(err)
        raise
    if np.any(kappa)<=0:
        raise NotImplementedError('The kappa parameter must be positive.')
        #sys.exit("The kappa parameter must be positive.")
    k = Sigma.shape[0]
    Omega = np.linalg.inv(Sigma) 
    ss = x - mu
    temp = np.sum(np.multiply(np.matmul(ss, Omega), ss ), axis=1)
    dens = (((np.log(k)+math.lgamma(k/2)) - ((k/2)*np.log(np.pi) +
          0.5*np.linalg.slogdet(Sigma)[1] + math.lgamma(1 + k/(2*kappa)) +
          (1 + k/(2*kappa))*np.log(2))) + kappa*(-0.5*temp) )[0]
    if not log:
        dens = np.exp(dens)
    return dens
    
def rmvpe(n, mu=np.zeros((1,2)), Sigma=np.diag([1,1]), kappa=1):
    mu = np.array(mu)
    k = mu.shape[1]
    if n > mu.shape[0]:
        mu = np.tile(mu, (int(np.ceil(n/mu.shape[0])), 1))[:n,]
        
    if k != Sigma.shape[0]: 
        raise NotImplementedError('mu and Sigma have non-conforming size.')
    
    evals, evecs = np.linalg.eig(Sigma)  #ev = eigen(Sigma, symmetric=TRUE)  
    if not all(evals>=-np.sqrt(np.finfo(float).eps)*abs(evals[0]) ):
        raise NotImplementedError("Sigma must be positive-definite.")
    
    SigmaSqrt = np.matmul(np.matmul(evecs, np.diag(evals)), evecs.T)
    radius = np.random.gamma(shape=k/(2*kappa), scale=1/2, size=n)**(1/(2*kappa))
    
    def runifsphere(n,k):
        
        if not isinstance(k, int):
            raise NotImplementedError('k must be an integer in [2,Inf).')
        if k<2:
            raise NotImplementedError('k must be an integer in [2,Inf).')
        Mnormal = np.array(np.random.normal(0, 1, n*k)).reshape(n, k, order='F')
        rownorms = np.sqrt(np.sum(Mnormal**2, axis=1))
        unifsphere = Mnormal/rownorms[:,np.newaxis]
        
        return unifsphere
    
    un = runifsphere(n=n, k=k)
    x = mu + radius[:,np.newaxis] * np.matmul(un, SigmaSqrt)
    
    return x