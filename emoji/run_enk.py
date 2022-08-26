"""
Main function to run ensemble Kalman (EnK) algorithms for Lorenz inverse problem
Shiwei Lan @ ASU, 2021
"""
 
# modules
import os,argparse,pickle
import numpy as np
import scipy.sparse as sps

# the inverse problem
from emoji import emoji
 
from joblib import Parallel, delayed
import multiprocessing

# EnK
import sys
sys.path.append( "../" )
from optimizer.EnK import *
 
np.set_printoptions(precision=3, suppress=True)
import warnings
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
# seed=2021
# np.random.seed(seed)


#Nicholas Haddad nehaddad@asu.edu ,Aariya Gage aariyagage@gmail.com
        
 
def main(seed=2021):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('mdlNO', nargs='?', type=int, default=0)
    parser.add_argument('ensemble_size', nargs='?', type=int, default=100)
    parser.add_argument('max_iter', nargs='?', type=int, default=2) #########################################50
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1])
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    parser.add_argument('mdls', nargs='?', type=str, default=('simple','STlik'))
    args = parser.parse_args()
    
    n_jobs = np.min([10, multiprocessing.cpu_count()])
    
    # set up random seed
    np.random.seed(seed)
    # define emoji Bayesian inverse problem
    spat_args={'basis_opt':'Fourier','l':1,'s':2,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    store_eig = True
    emj = emoji(spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)
    J,L = emj.prior.sz_t, spat_args['L']
    # initialization
    #u0:(L*J) = (2200*33) for vec mode, otherwise (128**2, 33)
    def sampleu(n=args.ensemble_size):
        # generate u with ensemble size
        u = [emj.prior.sample('fun').reshape((np.prod(emj.misfit.sz_x),emj.misfit.sz_t),order='F') for _ in range(n)]
        #u =[emj.prior.sample('vec') for _ in range(n)]
        return np.array(u)
    
    u0 = sampleu()
    ops_proj, obs_proj = emj.misfit.obs #(ops_proj,obs_proj) = emj.misfit.observe()[0]
    
    def run_enk(i=0):
    

        def G(u, i=0, ops_proj=ops_proj):
            '''
            only work with one time point i, u=(100,128**2)
            ops_proj = ops_proj[i]:(2170, 128**2), obs_proj = obs_proj[i]:(2170)
            '''
            
            G_slice = lambda i,u0: ops_proj[i].dot(u0)
            G_u = [G_slice(i,u0) for u0 in u]
            
            return np.array(G_u)
        
        G_ = lambda u,i=i: G(u,i)
        
        #y: (33, 2170)
        y=obs_proj[i]
        nz_cov=emj.misfit.nzcov
        
        data={'obs':y,'size':y.size,'cov':nz_cov.toarray()}
        prior={'mean':emj.prior.mean,'cov':emj.prior.bsv.tomat(),'sample':sampleu} 
        
        # EnK parameters
        nz_lvl=1.0
        err_thld=1e-2
     
        # run EnK to generate ensembles
        print("Preparing %s with step size %g for %s model..."
              % (args.algs[args.algNO],args.step_sizes[args.algNO],args.mdls[args.mdlNO]))
        enk = EnK(u0[:,:,i],G_,data,prior,stp_sz=args.step_sizes[args.algNO],nz_lvl=nz_lvl,err_thld=err_thld,
                  alg=args.algs[args.algNO],adpt=True)
        enk_fun=enk.run
        enk_args=(args.max_iter,True)
        savepath,filename=enk_fun(*enk_args)
        
        # append extra information including the count of solving
        filename_=os.path.join(savepath,filename+'.pckl')
        filename=os.path.join(savepath,'Emoji_'+filename+'_Time'+str(i)+'.pckl') # change filename
        os.rename(filename_, filename)
        f=open(filename,'ab')
        pickle.dump([ops_proj, obs_proj, i, args],f)
        f.close()
        
        
    
    
    #work for time point 0
    i=0
    run_enk(i)
    #error:ValueError: output array is read-only if Parallel
    #Parallel(n_jobs=n_jobs)(delayed(run_enk)(i) for i in range(J))
    
    
 
if __name__ == '__main__':
    
    np.random.seed(2022)
    main()
    
    
   