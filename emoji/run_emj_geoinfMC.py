"""
Main function to run Chen inverse problem to generate posterior samples
Shiwei Lan @ ASU, 2021
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from emoji import emoji

# MCMC
import sys
sys.path.append( "../" )
from sampler.geoinfMC import geoinfMC

np.set_printoptions(precision=3, suppress=True)
seed=2022
np.random.seed(seed)

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=2500)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.0001,.0001,.004,None,None]) # [.001,.005,.005] simple likelihood model
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()

    # define emoji Bayesian inverse problem
    spat_args={'basis_opt':'Fourier','l':1,'s':2,'q':1.0,'L':2000}
    temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
    store_eig = True
    emj = emoji(spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed, init_param=True)
    
    # initialization, here we sample from standard normal and transform
    unknown=np.random.randn(np.prod(emj.prior.sample().shape))  #emj.prior.sample()
    # MAP_file=os.path.join(os.getcwd(),'properties/MAP.pckl')
    # if os.path.isfile(MAP_file):
    #     f=open(MAP_file,'rb')
    #     unknown = pickle.load(f)
    #     f.close()
    # else:
    #     unknown=emj.get_MAP(SAVE=True)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    inf_GMC=geoinfMC(unknown,emj,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO])
  
    #inf_GMC.q=unknown[1:]; inf_GMC.dim=2
    geom_ord=[0]
    if any(s in args.algs[args.algNO] for s in ['MALA','HMC']): geom_ord.append(1)
    
    mc_fun=inf_GMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append ODE information including the count of solving
    filename_=os.path.join(inf_GMC.savepath,inf_GMC.filename+'.pckl')
    filename=os.path.join(inf_GMC.savepath,'emoji_'+inf_GMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    #soln_count=emj.ode.soln_count
    pickle.dump([args],f)
    f.close()
    
    # verify with load
    f=open(filename,'rb')
    mc_samp=pickle.load(f)
    #pde_info=pickle.load(f)
    enk_v = mc_samp[3].mean(axis=0)
    enk_f = emj.prior.vec2fun(enk_v).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    emj.misfit.plot_reconstruction(rcstr_imgs=enk_f, save_imgs=True, save_path='./reconstruction/'+
                                   args.algs[args.algNO]+'_num_samp'+str(args.num_samp)+'_step_sizes'+str(args.step_sizes[args.algNO]))
    f.close()


if __name__ == '__main__':
    main()