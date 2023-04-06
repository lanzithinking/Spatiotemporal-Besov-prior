"""
Plot estimates of uncertainty field u in linear inverse problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.ticker as mticker

# the inverse problem
from STEMPO import STEMPO


seed=2022
# define the inverse problem
data_args={'data_set':'simulation'}
spat_args={'basis_opt':'Fourier','l':.1,'s':1.0,'q':1.0,'L':2000}
temp_args={'ker_opt':'matern','l':.2,'q':1.0,'L':100}
store_eig = False
stpo = STEMPO(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
if stpo.misfit.data_set=='simulation':
    truth = np.rot90(stpo.misfit.truth,k=3,axes=(0,1))
else:
    truth = None
whiten = False

# models
pri_mdls=('q1p1','q2p2','iidT')
mdl_names=['STBP','STGP','time-independent']
num_mdls=len(pri_mdls)

# obtain estimates
folder = './reconstruction'
if os.path.exists(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl')):
    f=open(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl'),'rb')
    truth,maps,funs,errs=pickle.load(f)
    f.close()
    print('map_summary'+('_whiten' if whiten else '')+'.pckl has been read!')
else:
    maps=[]; funs=[]; errs=[]
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        fld_m = os.path.join(folder,'MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m])
        pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
        for f_i in pckl_files:
            try:
                f=open(os.path.join(fld_m,f_i),'rb')
                f_read=pickle.load(f)
                map_f=f_read[2]
                # map_f=f_read[2].swapaxes(0,1)
                # map_f=np.rot90(map_f,k=3,axes=(0,1))
                maps.append(map_f)
                funs.append(np.pad(f_read[3],(0,1000-len(f_read[3])),mode='constant',constant_values=np.nan))
                errs.append(np.pad(f_read[4],(0,1000-len(f_read[4])),mode='constant',constant_values=np.nan))
                f.close()
                print(f_i+' has been read!'); break
            except:
                pass
    maps=np.stack(maps)
    funs=np.stack(funs)
    errs=np.stack(errs)
    # save
    f=open(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl'),'wb')
    pickle.dump([truth,maps,funs,errs],f)
    f.close()

# # replot
# plt.rcParams['image.cmap'] = 'binary'
# for m in range(num_mdls):
#     map_flds = [f.path for f in os.scandir(folder) if 'MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m] in f.name]
#     for fld in map_flds:
#         map_f = maps[m]
#         stpo.misfit.plot_reconstruction(rcstr_imgs=map_f, save_imgs=True, save_path=fld)

# errors
N=1000
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    plt.plot(np.arange(1,N+1),{0:funs,1:errs}[i][:,:N].T)
    plt.yscale('log')
    ax.set_xlabel('iteration',fontsize=15)
    ax.set_ylabel({0:'negative posterior',1:'relative error'}[i],fontsize=15)
    ax.legend(labels=mdl_names,fontsize=14)
    ax.set_aspect('auto')
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
plt.subplots_adjust(wspace=0.2, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/map_errs'+('_whiten' if whiten else '')+'.png',bbox_inches='tight')
# plt.show()