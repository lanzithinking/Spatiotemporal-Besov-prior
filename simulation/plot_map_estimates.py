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
from simulation import simulation


seed=2022
# define the inverse problem
data_args={'n_x':2**8,'n_t':100,'nzvar':1e-2}
spat_args={'basis_opt':'Fourier','l':.1,'s':1.0,'q':1.0,'L':2000}
temp_args={'ker_opt':'matern','l':.1,'q':1.0,'L':100}
store_eig = False
sim = simulation(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
if hasattr(sim.misfit, 'truth'):
    truth = sim.misfit.truth
else:
    truth = None
whiten = True

# models
nxs=(2**4,2**5,2**7,2**8)
nts=(10,20,50,100)
num_nxs=len(nxs)
num_nts=len(nts)
pri_mdls=('q1','q2','iidT','pureBSV')
mdl_names=['STBP','STGP','time-uncorrelated','pure-Besov']
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
        fld_m = os.path.join(folder,'simulation_I65536_J100_/MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m])
        pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
        for f_i in pckl_files:
            try:
                f=open(os.path.join(fld_m,f_i),'rb')
                f_read=pickle.load(f)
                map_f=f_read[2]
                # map_f=f_read[2].swapaxes(0,1)
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
# plt.rcParams['image.cmap'] = 'gray'
# for m in range(num_mdls):
#     fld_m = os.path.join(folder,'simulation_I65536_J100_/MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m])
#     map_f = maps[m]
#     sim.misfit.plot_reconstruction(rcstr_imgs=map_f, save_imgs=True, save_path=fld_m, time_label=False)

# # errors
# if os.path.exists(os.path.join(folder,'map_rle'+('_whiten' if whiten else '')+'.pckl')):
#     f=open(os.path.join(folder,'map_rle'+('_whiten' if whiten else '')+'.pckl'),'rb')
#     truth,rle_m,rle_s,loglik_m,loglik_s,row_idx=pickle.load(f)
#     f.close()
#     print('map_rle'+('_whiten' if whiten else '')+'.pckl has been read!')
#     rle_m=rle_m.reshape((num_nxs,num_nts,num_mdls))
#     rle_s=rle_s.reshape((num_nxs,num_nts,num_mdls))
#
#     # plot relative errors
#     fig = plt.figure(figsize=(10,5))
#     for m in range(num_mdls):
#         for i in range(num_nxs):
#             plt.errorbar(np.arange(num_nts),rle_m[i,:,m], yerr=rle_s[i,:,m])#, label=mdl_names[m])
#     plt.yscale('log')
#     plt.set_xlabel('J',fontsize=15)
#     plt.set_ylabel('Relative Error',fontsize=15)
#     # plt.legend(labels=mdl_names,fontsize=14)
#     # plt.legend(loc='upper right',fontsize=14)
#     plt.set_aspect('auto')
#     plt.yaxis.set_minor_formatter(mticker.ScalarFormatter())
#     # save plot
#     # fig.tight_layout()
#     # plt.savefig(folder+'/map_errs'+('_whiten' if whiten else '')+'.png',bbox_inches='tight')
#     plt.show()