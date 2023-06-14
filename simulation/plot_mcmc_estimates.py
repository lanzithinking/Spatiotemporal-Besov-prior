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


# algorithms
algs=[16, 64, 256, 1024, 4096] #('wpCN','winfMALA','winfHMC','winfmMALA','winfmHMC','ESS')
alg_names=list(map(str, n_x_list))
num_algs=len(n_x_list)
# obtain estimates
folder = './result'
if os.path.exists(os.path.join(folder,'mcmc_summary.npz')):
    med_f,mean_f,std_f,truth_l=pickle.load(os.path.join(folder,'mcmc_summary.npz'))
    print('mcmc_summary.npz has been read!')
else:
    med_f=[[]]*num_algs
    mean_f=[[]]*num_algs
    std_f=[[]]*num_algs
    truth_l=[[]]*num_algs
    obs_l=[[]]*num_algs
    npz_files=[f for f in os.listdir(folder) if f.endswith('.npz')]
    for i in range(num_algs):
        # preparation
        print('Processing '+alg_names[i]+' algorithm...\n')
        data_args['n_x']=n_x_list[i]
        if data_args['n_x']>=spat_args['L']:                                                
            data_args['space']='vec' 
        sml = simulation(**data_args,spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)
        truth_l[i], obs_l[i] = sml.misfit.truth.reshape(np.append(sml.misfit.n_x,sml.misfit.n_t),order='F'), sml.misfit.obs.reshape(np.append(sml.misfit.n_x,sml.misfit.n_t),order='F')
        # get estimates
        for f_i in npz_files:
            if (('_dim'+str(algs[i]*sml.prior.sz_t)+'_' in f_i)  and (data_args['space']=='fun')) or (('_dim'+str(spat_args['L']*sml.prior.sz_t)+'_' in f_i) and (data_args['space']=='vec')):  
                try:
                    f_read=np.load(os.path.join(folder,f_i))
                    samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                    try:
                        if sml.prior.space=='vec': samp=sml.prior.vec2fun(samp.T).T
                        med_f[i]=np.median(samp,axis=0).reshape(np.append(sml.misfit.n_x,sml.misfit.n_t),order='F')
                        mean_f[i]=np.mean(samp,axis=0).reshape(np.append(sml.misfit.n_x,sml.misfit.n_t),order='F')
                        std_f[i]=np.std(samp,axis=0).reshape(np.append(sml.misfit.n_x,sml.misfit.n_t),order='F')
                    except Exception as e:
                        print(e)
                        mean_f[i]=0; std_f[i]=0
                        n_samp=samp.shape[0]
                        for s in range(n_samp):
                            samp_s=sml.prior.vec2fun(samp[s]) if sml.prior.space=='vec' else samp[s]
                            mean_f[i]+=samp_s/n_samp
                            std_f[i]+=samp_s**2/n_samp
                        std_f[i]=np.sqrt(std_f[i]-mean_f[i]**2)
                        mean_f[i]=mean_f[i].reshape(np.append(sml.misfit.n_x,sml.misfit.n_t),order='F')
                        std_f[i]=std_f[i].reshape(np.append(sml.misfit.n_x,sml.misfit.n_t),order='F')
                        # med_f[i]=None
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    np.savez_compressed(os.path.join(folder,'mcmc_summary.npz'),med_f=med_f,mean_f=mean_f,std_f=std_f,truth_l=truth_l,obs_l=obs_l)


# plot 
# plt.rcParams['image.cmap'] = 'jet'
num_rows=1
# posterior mean/median with credible band
fig,axes = plt.subplots(nrows=num_rows,ncols=3,sharex=True,sharey=True,figsize=(16,5))
titles = ['16', '256', '4096']
times = np.linspace(0,2,sml.misfit.n_t)
loc = 4

for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    ax.plot(times, truth_l[2*i][loc])
    ax.scatter(times, obs_l[2*i][loc], color='orange')
    ax.plot(times, mean_f[2*i][loc], linewidth=2, linestyle='--', color='red')
    ax.fill_between(times,mean_f[2*i][loc]-1.96*std_f[2*i][loc],mean_f[2*i][loc]+1.96*std_f[2*i][loc],color='b',alpha=.2)
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
    
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/sml_'+sml.misfit.truth_name+'_mcmc_estimates_comparepri.png',bbox_inches='tight')
# plt.show()

'''
# plot
for i in range(num_algs):
    if len(med_f[i])!=0:
        sml.misfit.plot_reconstruction(rcstr_imgs=med_f[i], save_imgs=True, save_path=folder+'/reconstruction/'+alg_names[i]+'_median')
    if len(mean_f[i])!=0:
        sml.misfit.plot_reconstruction(rcstr_imgs=mean_f[i], save_imgs=True, save_path=folder+'/reconstruction/'+alg_names[i]+'_mean')
    if len(std_f[i])!=0:
        sml.misfit.plot_reconstruction(rcstr_imgs=std_f[i], save_imgs=True, save_path=folder+'/reconstruction/'+alg_names[i]+'_std')
'''