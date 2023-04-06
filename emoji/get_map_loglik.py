"""
Get root mean squared error (RMSE) of MAP for the process u in the time series problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
# from emoji import emoji
from misfit import misfit

seed=2022
# define the inverse problem
data_args={'data_set':'30proj','data_thinning':3}
# spat_args={'basis_opt':'Fourier','l':.1,'s':1.0,'q':1.0,'L':2000}
# temp_args={'ker_opt':'matern','l':.5,'q':1.0,'L':100}
# store_eig = True
# emj = emoji(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
msft = misfit(**data_args)
whiten = True

# models
pri_mdls=('q1p1','q2p2','iidT')
mdl_names=['STBP','STGP','time-independent']
num_mdls=len(pri_mdls)
# store results
loglik=np.zeros(num_mdls)
# obtain estimates
folder = './reconstruction'
if os.path.exists(os.path.join(folder,'map_loglik'+('_whiten' if whiten else '')+'.pckl')):
    f=open(os.path.join(folder,'map_loglik'+('_whiten' if whiten else '')+'.pckl'),'rb')
    loglik=pickle.load(f)
    f.close()
    print('map_loglik'+('_whiten' if whiten else '')+'.pckl has been read!')
else:
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        map_flds = [f.path for f in os.scandir(folder) if 'MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m] in f.name]
        for fld in map_flds:
            pckl_files=[f for f in os.listdir(fld) if f.endswith('.pckl')]
            for k in pckl_files:
                f=open(os.path.join(fld,k),'rb')
                f_read=pickle.load(f)
                map=f_read[2]
                loglik[m]=-msft.cost(map)
                f.close()
                print(k+' has been read!'); break
    # save
    f=open(os.path.join(folder,'map_loglik'+('_whiten' if whiten else '')+'.pckl'),'wb')
    pickle.dump([loglik],f)
    f.close()

# save
import pandas as pd
sumry = pd.DataFrame(data=np.vstack((loglik,)),columns=mdl_names[:num_mdls],index=['log-lik'])
sumry.to_csv(os.path.join(folder,'map_loglik'+('_whiten' if whiten else '')+'.csv'),columns=mdl_names[:num_mdls])