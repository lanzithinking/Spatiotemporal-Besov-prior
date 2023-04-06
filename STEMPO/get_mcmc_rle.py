"""
Get relative error (rle) and root mean square error (RMSE) for uncertainty field u in linear inverse problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from STEMPO import STEMPO

seed=2022
# define the inverse problem
data_args={'data_set':'simulation'}
spat_args={'basis_opt':'Fourier','l':1,'s':1,'q':1.0,'L':2000}
temp_args={'ker_opt':'matern','l':.5,'s':2,'q':1.0,'L':100}
store_eig = False
# stpo = STEMPO(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)

# models
pri_mdls=('q1p1','q2p2')#,'iidT')
mdl_names=['STBP','STGP']#,'time-independent']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './analysis'
if os.path.exists(os.path.join(folder,'rle_summary.npz')):
    rle_m,rle_s,rmse_m,rmse_s=load(os.path.join(folder,'rle_summary.npz'))
    print('rle_summary.npz has been read!')
else:
    # store results
    rle_m=np.zeros(num_mdls)
    rle_s=np.zeros(num_mdls)
    rmse_m=np.zeros(num_mdls)
    rmse_s=np.zeros(num_mdls)
    for m in range(num_mdls):
        # preparation
        spat_args['q']={'q1p1':1,'q2p2':2}[pri_mdls[m]]
        temp_args['q']={'q1p1':1,'q2p2':2}[pri_mdls[m]]
        stpo = STEMPO(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)
        stpo.misfit.nzcov = sps.eye(stpo.misfit.nzcov.shape[0], format='csr')
        if stpo.misfit.data_set=='simulation':
            truth = stpo.misfit.truth
        else:
            truth = None
        print('Processing '+pri_mdls[m]+' prior model...\n')
        npz_files=[f for f in os.listdir(folder) if f.endswith('.npz')]
        # get estimates
        for f_i in npz_files:
            if '_'+pri_mdls[m] in f_i:
                try:
                    f_read=np.load(os.path.join(folder,f_i))
                    samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                    n_samp=samp.shape[0]
                    for s in range(n_samp):
                        if stpo.misfit.data_set=='simulation':
                            rle_s=np.linalg.norm(samp[s]-truth.flatten(order='F'))/np.linalg.norm(truth)
                            rle_m[m]+=rle_s/n_samp
                            rle_s[m]+=rle_s**2/n_samp
                        mse_s=stpo._get_misfit(samp[s])*2
                        rmse_m[m]+=np.sqrt(mse_s)/n_samp
                        rmse_s[m]+=mse_s/n_samp
                    rle_s[m]=np.sqrt(rle_s[m]-rle_m[m]**2)
                    rmse_s[m]=np.sqrt(rmse_s[m]-rmse_m[m]**2)
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    np.savez_compressed(os.path.join(folder,'rle_summary.npz'),rle_m=rle_m,rle_s=rle_s,rmse_m=rmse_m,rmse_s=rmse_s)

# save
import pandas as pd
means = pd.DataFrame(data=np.vstack((rle_m,rmse_m)),columns=mdl_names[:num_mdls],index=['relative-error','RMSE'])
stds = pd.DataFrame(data=np.vstack((rle_s,rmse_s)),columns=mdl_names[:num_mdls],index=['relative-error','RMSE'])
means.to_csv(os.path.join(folder,'rle_means.csv'),columns=mdl_names[:num_mdls])
stds.to_csv(os.path.join(folder,'rle_stds.csv'),columns=mdl_names[:num_mdls])