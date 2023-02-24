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
algs=('wpCN','winfMALA','winfHMC','winfmMALA','winfmHMC','ESS')
alg_names=('wpCN','w$\infty$-MALA','w$\infty$-HMC','w$\infty$-mMALA','w$\infty$-mHMC','ESS')
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
            if algs[i]+'_' in f_i:
                try:
                    f_read=np.load(os.path.join(folder,f_i))
                    samp=f_read['samp']
                    try:
                        if emj.prior.space=='vec': samp=emj.prior.vec2fun(samp.T).T
                        med_f[i]=np.median(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
                        mean_f[i]=np.mean(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
                        std_f[i]=np.std(samp,axis=0).reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
                    except Exception as e:
                        print(e)
                        mean_f=0; std_f=0
                        n_samp=samp.shape[0]
                        for i in range(n_samp):
                            samp_i=emj.prior.vec2fun(samp[i]) if emj.prior.space=='vec' else samp[i]
                            mean_f+=samp_i/n_samp
                            std_f+=samp_i**2/n_samp
                        std_f=np.sqrt(std_f-mean_f**2)
                        mean_f=mean_f.reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
                        std_f=std_f.reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
                        # med_f=None
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    np.savez_compressed(os.path.join(folder,'mcmc_summary.npz'),med_f=med_f,mean_f=mean_f,std_f=std_f)

# plot
for i in range(num_algs):
    if len(med_f[i])!=0:
        emj.misfit.plot_reconstruction(rcstr_imgs=med_f[i], save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_median')
    if len(mean_f[i])!=0:
        emj.misfit.plot_reconstruction(rcstr_imgs=mean_f[i], save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_mean')
    if len(std_f[i])!=0:
        emj.misfit.plot_reconstruction(rcstr_imgs=std_f[i], save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_std')