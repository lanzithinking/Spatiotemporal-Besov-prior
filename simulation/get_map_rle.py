"""
Get root mean squared error (RMSE) of MAP for the process u in the annulus problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from misfit import misfit

seed=2022
# define the inverse problem
# data_args={'n_x':2**4,'n_t':100,'nzvar':1e-3}
# # spat_args={'basis_opt':'Fourier','l':.1,'s':1.0,'q':1.0,'L':2000}
# # temp_args={'ker_opt':'matern','l':.2,'q':1.0,'L':100}
# # store_eig = True
# # stpo = simulation(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)
# msft = misfit(**data_args)
# if hasattr(msft, 'truth'):
#     truth = msft.truth
# else:
#     truth = None
whiten = True

# models
nxs=(2**4,2**5,2**7,2**8)
nts=(10,20,50,100)
num_nxs=len(nxs)
num_nts=len(nts)
pri_mdls=('q1','q2','iidT')
mdl_names=['STBP','STGP','time-independent']
num_mdls=len(pri_mdls)
row_idx=[]
# obtain estimates
folder = './reconstruction'
if not os.path.exists(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl')):
    maps=[[]]*num_mdls
    funs=[[[]]*num_mdls]*num_nxs*num_nts
    errs=[[[]]*num_mdls]*num_nxs*num_nts
    truth=[[]]*(num_nxs*num_nts)
if os.path.exists(os.path.join(folder,'map_rle'+('_whiten' if whiten else '')+'.pckl')):
    f=open(os.path.join(folder,'map_rle'+('_whiten' if whiten else '')+'.pckl'),'rb')
    truth,rle_m,rle_s,loglik_m,loglik_s=pickle.load(f)
    f.close()
    print('map_rle'+('_whiten' if whiten else '')+'.pckl has been read!')
else:
    # store results
    rle_m=np.zeros((num_nxs*num_nts, num_mdls))
    rle_s=np.zeros((num_nxs*num_nts, num_mdls))
    loglik_m=np.zeros((num_nxs*num_nts, num_mdls))
    loglik_s=np.zeros((num_nxs*num_nts, num_mdls))
    for i in range(num_nxs):
        for j in range(num_nts):
            data_args={'n_x':nxs[i],'n_t':nts[j]}
            msft = misfit(**data_args)
            row_idx.append('nx{}_nt{}_'.format(nxs[i], nts[j]))
            for m in range(num_mdls):
                print('Processing simulation I{}_J{}_'.format(nxs[i]**2, nts[j])+' for '+ pri_mdls[m]+' prior model...\n')
                rle=[]; loglik=[]
                num_read=0
                fld_m = os.path.join(folder,'simulation_I{}_J{}_'.format(nxs[i]**2, nts[j]))+'/MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m]#+'_repeat')
                pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
                for f_i in pckl_files:
                    try:
                        f=open(os.path.join(fld_m,f_i),'rb')
                        f_read=pickle.load(f)
                        map=f_read[-3]
                        if hasattr(msft, 'truth'):
                            # rle.append(np.linalg.norm(map-msft.truth)/np.linalg.norm(msft.truth))
                            rle.append(np.linalg.norm((map-msft.truth).reshape((-1,msft.sz_t),order='F'),np.inf)/np.linalg.norm(msft.truth.reshape((-1,msft.sz_t),order='F'),np.inf))
                            if not truth[i*num_nts+j]: truth[i*num_nts+j].append(msft.truth)
                        loglik.append(-msft.cost(map))
                        num_read+=1
                        f.close()
                        print(f_i+' has been read!')
                    except Exception as e:
                        print(e); pass
                print('%d experiment(s) have been processed for %s prior model.' % (num_read, pri_mdls[m]))
                if num_read>0:
                    rle_m[i*num_nts+j,m] = np.median(rle)
                    rle_s[i*num_nts+j,m] = np.std(rle)
                    loglik_m[i*num_nts+j,m] = np.median(loglik)
                    loglik_s[i*num_nts+j,m] = np.std(loglik)
                    # get the best for plotting
                    if not os.path.exists(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl')):
                        f_i=pckl_files[np.argmin(rle)]
                        f=open(os.path.join(fld_m,f_i),'rb')
                        f_read=pickle.load(f)
                        map_f=f_read[-3]
                        if i==num_nxs-1 and j==num_nts-1: maps[m]=map_f
                        funs[i*num_nts+j][m]=np.pad(f_read[-2],(0,1000-len(f_read[-2])),mode='constant',constant_values=np.nan)
                        errs[i*num_nts+j][m]=np.pad(f_read[-1],(0,1000-len(f_read[-1])),mode='constant',constant_values=np.nan)
                        f.close()
                        print(f_i+' has been selected to print!')
                        msft.plot_reconstruction(rcstr_imgs=map_f, save_imgs=True, save_path=fld_m, time_label=False, anim=False)
    # save
    f=open(os.path.join(folder,'map_rle'+('_whiten' if whiten else '')+'.pckl'),'wb')
    pickle.dump([truth,rle_m,rle_s,loglik_m,loglik_s,row_idx],f)
    f.close()
if not os.path.exists(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl')):
    maps=np.stack(maps)
    # funs=np.stack(funs)
    # errs=np.stack(errs)
    # save
    f=open(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl'),'wb')
    pickle.dump([truth,maps,funs,errs],f)
    f.close()

# save
import pandas as pd
means = pd.DataFrame(data=rle_m,columns=mdl_names[:num_mdls],index=row_idx)
stds = pd.DataFrame(data=rle_s,columns=mdl_names[:num_mdls],index=row_idx)
means.to_csv(os.path.join(folder,'map_rle_means'+('_whiten' if whiten else '')+'.csv'),columns=mdl_names[:num_mdls])
stds.to_csv(os.path.join(folder,'map_rle_stds'+('_whiten' if whiten else '')+'.csv'),columns=mdl_names[:num_mdls])