"""
Plot estimates of uncertainty field u in linear inverse problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from emoji import emoji


seed=2022
# define the inverse problem
data_args={'data_set':'60proj','data_thinning':2}
spat_args={'basis_opt':'Fourier','l':1,'s':1,'q':1.0,'L':2000}
# spat_args={'basis_opt':'wavelet','wvlet_typ':'Meyer','l':1,'s':2,'q':1.0,'L':2000}
# temp_args={'ker_opt':'powexp','l':.5,'s':2,'q':1.0,'L':100}
temp_args={'ker_opt':'matern','l':.5,'s':2,'q':1.0,'L':100}
store_eig = True
emj = emoji(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)

# algorithms
algs=('pCN','infMALA','infHMC','infmMALA','infmHMC','ESS')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','$\infty$-mMALA','$\infty$-mHMC','ESS')
num_algs=len(algs)
# obtain estimates
folder = './analysis'
if os.path.exists(os.path.join(folder,'mcmc_summary.npz')):
    med_f,mean_f,std_f=load(os.path.join(folder,'mcmc_summary.npz'))
    print('mcmc_summary.npz has been read!')
else:
    med_f=[[]]*num_algs
    mean_f=[[]]*num_algs
    std_f=[[]]*num_algs
    npz_files=[f for f in os.listdir(folder) if f.endswith('.npz')]
    for i in range(num_algs):
        # preparation
        print('Processing '+algs[i]+' algorithm...\n')
        # get estimates
        for f_i in npz_files:
            if '_'+algs[i]+'_' in f_i:
                try:
                    f_read=np.load(os.path.join(fld_i,f_i))
                    samp=f_read[-4]
                    if emj.prior.space=='vec': samp=emj.prior.vec2fun(samp.T).T
                    med_f[i]=np.median(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
                    mean_f[i]=np.mean(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
                    std_f[i]=np.std(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    np.savez_compressed(os.path.join(folder,'mcmc_summary.npz'),med_f=med_f,mean_f=mean_f,std_f=std_f)

# plot
for i in range(num_algs):
    emj.misfit.plot_reconstruction(rcstr_imgs=med_f, save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_median')
    emj.misfit.plot_reconstruction(rcstr_imgs=mean_f, save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_mean')
    emj.misfit.plot_reconstruction(rcstr_imgs=std_f, save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_std')