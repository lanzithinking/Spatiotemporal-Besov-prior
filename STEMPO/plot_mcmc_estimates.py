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
from STEMPO import STEMPO


seed=2022
# define the inverse problem
data_args={'data_set':'simulation'}
spat_args={'basis_opt':'Fourier','l':.1,'s':1,'q':1.0,'L':2000}
temp_args={'ker_opt':'matern','l':.5,'s':2,'q':1.0,'L':100}
store_eig = True
stpo = STEMPO(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)

# algorithms
algs=('winfmMALA',)#'winfmHMC','ESS')
alg_names=('w$\infty$-mMALA',)#'w$\infty$-mHMC','ESS')
num_algs=len(algs)
pri_mdls=('q1p1','q2p2')
mdl_names=['STBP','STGP']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './analysis'
if os.path.exists(os.path.join(folder,'mcmc_summary.npz')):
    f_read=np.load(os.path.join(folder,'mcmc_summary.npz'),allow_pickle=True)
    med_f,mean_f,std_f=f_read['med_f'],f_read['mean_f'],f_read['std_f']
    print('mcmc_summary.npz has been read!')
else:
    med_f=[[]]*num_algs*num_mdls
    mean_f=[[]]*num_algs*num_mdls
    std_f=[[]]*num_algs*num_mdls
    npz_files=[f for f in os.listdir(folder) if f.endswith('.npz')]
    for i in range(num_algs):
        # preparation
        print('Processing '+algs[i]+' algorithm...\n')
        for j in range(num_mdls):
            spat_args['q']={'q1p1':1,'q2p2':2}[pri_mdls[j]]
            temp_args['q']={'q1p1':1,'q2p2':2}[pri_mdls[j]]
            stpo = STEMPO(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)
            # get estimates
            for f_i in npz_files:
                if algs[i]+'_' in f_i and '_'+pri_mdls[j]+'_' in f_i:
                    try:
                        f_read=np.load(os.path.join(folder,f_i))
                        samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                        # try:
                        #     if stpo.prior.space=='vec': samp=stpo.prior.vec2fun(samp.T).T
                        #     med_f[i*num_mdls+j]=np.rot90(np.median(samp,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
                        #     mean_f[i*num_mdls+j]=np.rot90(np.mean(samp,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
                        #     std_f[i*num_mdls+j]=np.rot90(np.std(samp,axis=0).reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
                        # except Exception as e:
                        #     print(e)
                        mean_f[i*num_mdls+j]=0; std_f[i*num_mdls+j]=0
                        n_samp=samp.shape[0]
                        for s in range(n_samp):
                            samp_s=stpo.prior.vec2fun(samp[s]) if stpo.prior.space=='vec' else samp[s]
                            mean_f[i*num_mdls+j]+=samp_s/n_samp
                            std_f[i*num_mdls+j]+=samp_s**2/n_samp
                        std_f[i*num_mdls+j]=np.sqrt(std_f[i*num_mdls+j]-mean_f[i*num_mdls+j]**2)
                        mean_f[i*num_mdls+j]=np.rot90(mean_f[i*num_mdls+j].reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
                        std_f[i*num_mdls+j]=np.rot90(std_f[i*num_mdls+j].reshape(np.append(stpo.misfit.sz_x,stpo.misfit.sz_t),order='F'),k=3,axes=(0,1))
                        # med_f[i*num_mdls+j]=None
                        print(f_i+' has been read!'); break
                    except:
                        pass
    # save
    np.savez_compressed(os.path.join(folder,'mcmc_summary.npz'),med_f=med_f,mean_f=mean_f,std_f=std_f)

# plot
for i in range(num_algs):
    for j in range(num_mdls):
        if len(med_f[i*num_mdls+j])!=0:
            stpo.misfit.plot_reconstruction(rcstr_imgs=med_f[i*num_mdls+j], save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_median_'+pri_mdls[j], time_label=False)
        if len(mean_f[i*num_mdls+j])!=0:
            stpo.misfit.plot_reconstruction(rcstr_imgs=mean_f[i*num_mdls+j], save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_mean_'+pri_mdls[j], time_label=False)
        if len(std_f[i*num_mdls+j])!=0:
            stpo.misfit.plot_reconstruction(rcstr_imgs=std_f[i*num_mdls+j], save_imgs=True, save_path=folder+'/reconstruction/'+algs[i]+'_std_'+pri_mdls[j], time_label=False)