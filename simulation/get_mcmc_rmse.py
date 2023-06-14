"""
Get root mean square error (RMSE) for uncertainty field u in linear inverse problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from simulation import simulation

seed=2022

n_x_list = [16, 64, 256, 1024, 4096]
# define emoji Bayesian inverse problem
data_args={'truth_option':1,'n_x':n_x_list[0], 'space':'fun'}
spat_args={'basis_opt':'Fourier','l':.1,'s':1,'q':1,'L':2000}
# spat_args={'basis_opt':'wavelet','wvlet_typ':'Meyer','l':1,'s':2,'q':args.q,'L':2000}
# temp_args={'ker_opt':'powexp','l':.5,'s':2,'q':1.0,'L':100}
temp_args={'ker_opt':'matern','l':.5,'s':2,'q':1.0,'L':100}
store_eig = True
#sml = simulation(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)



# models

alg_names=list(map(str, n_x_list))
num_mdls=len(n_x_list)
# obtain estimates
folder = './result'
if os.path.exists(os.path.join(folder,'rmse_summary.npz')):
    rmse_m,rmse_s=np.load(os.path.join(folder,'rmse_summary.npz'))
    print('rmse_summary.npz has been read!')
else:
    # store results
    rmse_m=np.zeros(num_mdls)
    rmse_s=np.zeros(num_mdls)
    for m in range(num_mdls):
        # preparation
        data_args['n_x']=n_x_list[m]
        if data_args['n_x']>=spat_args['L']:                                                
            data_args['space']='vec' 
        sml = simulation(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)
        #sml.misfit.nzcov = sps.eye(sml.misfit.nzcov.shape[0], format='csr')
        print('Processing '+alg_names[m]+' prior model...\n')
        npz_files=[f for f in os.listdir(folder) if f.endswith('.npz')]
        # get estimates
        for f_i in npz_files:
            if (('_dim'+str(n_x_list[m]*sml.prior.sz_t)+'_' in f_i) and (data_args['space']=='fun') ) or (('_dim'+str(spat_args['L']*sml.prior.sz_t)+'_' in f_i) and (data_args['space']=='vec')):
                try:
                    f_read=np.load(os.path.join(folder,f_i))
                    samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                    n_samp=samp.shape[0]
                    for s in range(n_samp):
                        mse_s=sml._get_misfit(samp[s])*2
                        rmse_m[m]+=np.sqrt(mse_s)/n_samp
                        rmse_s[m]+=mse_s/n_samp
                    rmse_s[m]=np.sqrt(rmse_s[m]-rmse_m[m]**2)
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    np.savez_compressed(os.path.join(folder,'rmse_summary.npz'),rmse_m=rmse_m,rmse_s=rmse_s)

# save
import pandas as pd
rmse_m = pd.DataFrame(data=rmse_m[None,:],columns=alg_names[:num_mdls])
rmse_s = pd.DataFrame(data=rmse_s[None,:],columns=alg_names[:num_mdls])
rmse_m.to_csv(os.path.join(folder,'RMSE_mean.csv'),columns=alg_names[:num_mdls])
rmse_s.to_csv(os.path.join(folder,'RMSE_std.csv'),columns=alg_names[:num_mdls])