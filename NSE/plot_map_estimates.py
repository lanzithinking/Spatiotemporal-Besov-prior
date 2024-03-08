"""
Plot estimates of uncertainty field u in non-linear inverse problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.ticker as mticker

# the inverse problem
from NSE import *


seed=2022
# define the inverse problem
data_args={'data_set':'V1e-3','data_thinning':4}
spat_args={'basis_opt':'Fourier','l':.1,'s':1,'q':1.0,'L':2000}
temp_args={'ker_opt':'matern','l':.5,'s':2,'q':1.0,'L':100}
store_eig = False
nse = NSE(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
truth = nse.misfit.truth
whiten = True

# models
pri_mdls=('q1p1','q2p2','iidT')
mdl_names=['STBP','STGP','time-uncorrelated']
num_mdls=len(pri_mdls)

# obtain estimates
folder = './invsol'
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
                map_f=f_read[-3]
                maps.append(map_f)
                funs.append(np.pad(f_read[-2],(0,1000-len(f_read[-2])),mode='constant',constant_values=np.nan))
                errs.append(np.pad(f_read[-1],(0,1000-len(f_read[-1])),mode='constant',constant_values=np.nan))
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

# replot
# plt.rcParams['image.cmap'] = 'binary'
# for m in range(num_mdls):
#     fld_m = os.path.join(folder,'MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m])
#     map_f = maps[m]
#     nse.misfit.plot_data(dat_imgs=map_f, save_imgs=True, save_path=fld_m, time_label=False)

# selective plot
# import sys
# sys.path.append( "../" )
# from util.common_colorbar import common_colorbar
# times = [0,3,6,9]
# plt.rcParams['image.cmap'] = 'jet'
# for m in range(num_mdls):
#     fld_m = os.path.join(folder,'MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m])
#     map_f = maps[m]
#     fig,axes = plt.subplots(nrows=1,ncols=len(times),sharex=True,sharey=True,figsize=(18,4))
#     sub_figs = [None]*len(axes.flat)
#     for i,ax in enumerate(axes.flat):
#         plt.axes(ax)
#         sub_figs[i]=ax.imshow(map_f[:,:,times[i]], origin='lower', extent=[0,1,0,1])
#         ax.tick_params(axis='x', labelsize=16)
#         ax.tick_params(axis='y', labelsize=16)
#         ax.set_title('t = '+str(times[i]),fontsize=18)
#     plt.subplots_adjust(wspace=0.1, hspace=0.2)
#     fig=common_colorbar(fig,axes,sub_figs)
#     plt.savefig(fld_m+'/NSE_map'+('_whiten' if whiten else '')+'.png',bbox_inches='tight')

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