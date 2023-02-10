"""
Main function to run ensemble Kalman (EnK) algorithms for the dynamic linear inverse problem of crossPhantom.
Shiwei Lan @ ASU, 2022
"""
 
# modules
import os,argparse,pickle
import numpy as np
import scipy.sparse as sps

# the inverse problem
from crossPhantom import crossPhantom
 
from joblib import Parallel, delayed
import multiprocessing

# EnK
import sys
sys.path.append( "../" )
from optimizer.EnK_st import *
 
np.set_printoptions(precision=3, suppress=True)
import warnings
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

def main(seed=2022):
    #i: which time point do we want to work on (33, 0-32)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('ensemble_size', nargs='?', type=int, default=100)
    parser.add_argument('max_iter', nargs='?', type=int, default=50) 
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1])
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    args = parser.parse_args()
    
    # set up random seed
    np.random.seed(seed)
    # define emoji Bayesian inverse problem
    spat_args={'basis_opt':'Fourier','l':1,'s':2,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    store_eig = True
    crsptm = crossPhantom(spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed, init_param=True)
    
    # initialization
    u0 = np.stack([crsptm.init_parameter + .1*crsptm.prior.sample() for _ in range(args.ensemble_size)]) # (esmblsz, LJ), L=200, J=33
    
    # Au = lambda u: np.stack([crsptm.misfit.obs[0][t].dot(u_t) for t,u_t in enumerate(u.reshape(crsptm.misfit.sz_t,-1))]).T
    # if args.ensemble_size>200:
    #     n_jobs = np.min([10, multiprocessing.cpu_count()])
    #     G = lambda u: np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j: Au(crsptm.prior.vec2fun(u_j)))(u_j) for u_j in u))
    # else:
    #     G = lambda u: np.stack([Au(crsptm.prior.vec2fun(u_j)) for u_j in u])
    # import time
    # t0=time.time()
    # res0=G(u0)
    # t1=time.time()
    # print('Time used is %.4f' % (t1-t0))
    
    eigv, eigf = crsptm.prior.bsv.eigs() # (I,L)
    tA = [crsptm.misfit.obs[0][t].dot(eigf) for t in range(crsptm.misfit.sz_t)] # each of size (m=2170,L=2000)
    tAu = lambda u: np.stack([tA[t].dot(u_t) for t,u_t in enumerate(u.reshape(crsptm.misfit.sz_t,-1))]).T
    if args.ensemble_size>200:
        n_jobs = np.min([10, multiprocessing.cpu_count()])
        G = lambda u: np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j: tAu(u_j))(u_j) for u_j in u))
    else:
        G = lambda u: np.stack([tAu(u_j) for u_j in u])
    # t0=time.time()
    # res1=G(u0)
    # t1=time.time()
    # print('Time used is %.4f' % (t1-t0))
    # print('Difference: %.6f' % (abs(res1-res0).max()))
        
    crsptm.misfit.data = np.stack(crsptm.misfit.obs[1]).T
    crsptm.prior.cov = sps.spdiags(np.tile(eigv,crsptm.prior.sz_t),0,crsptm.prior.L*crsptm.prior.sz_t,crsptm.prior.L*crsptm.prior.sz_t)
    
    # EnK parameters
    nz_lvl=0.1
    err_thld=1e-2
 
    # run EnK to generate ensembles
    print("Preparing %s with %d ensembles for step size %g ..."
          % (args.algs[args.algNO],args.ensemble_size,args.step_sizes[args.algNO]))
    enk = EnK(u0,G,crsptm.misfit,crsptm.prior,stp_sz=args.step_sizes[args.algNO],nz_lvl=nz_lvl,err_thld=err_thld,
              alg=args.algs[args.algNO], reg=False, adpt=True)
    enk_fun=enk.run
    enk_args=(args.max_iter,True)
    savepath,filename=enk_fun(*enk_args)
    
    # append extra information including the count of solving
    filepath_=os.path.join(savepath,filename+'.pckl')
    filepath=os.path.join(savepath,'CrossPhantom_'+filename+'.pckl') # change filename
    os.rename(filepath_, filepath)
    f=open(filepath,'ab')
    pickle.dump([crsptm.misfit.data, crsptm.prior.cov, args],f)
    f.close()
    
    # plot
    f=open(filepath,'rb')
    loaded=pickle.load(f)
    est, err=loaded[:2]
    f.close()
    enk_v=est[np.argmin(err)]
    enk_f = crsptm.prior.vec2fun(enk_v).reshape(np.append(crsptm.misfit.sz_x,crsptm.misfit.sz_t),order='F').swapaxes(0,1)
    crsptm.misfit.plot_reconstruction(rcstr_imgs=enk_f, save_imgs=True, save_path='./reconstruction/'+args.algs[args.algNO]+'_ensbl'+str(args.ensemble_size))

if __name__ == '__main__':
    seed = 2022
    np.random.seed(seed)
    main(seed)
