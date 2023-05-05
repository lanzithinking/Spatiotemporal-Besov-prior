"""
Get root mean squared error (RMSE) of MAP for the process u in the time series problem.
----------------------
Shiwei Lan @ ASU, 2022
"""
#pip3 install scikit-image
import os,pickle,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
# from NSE import *
from misfit import misfit
sys.path.append( "../" )
from NSE.haar_psi import haar_psi_numpy

def PSNR(reco, gt):
    mse = np.mean((np.asarray(reco) - gt)**2)
    if mse == 0.:
        return float('inf')
    data_range = (np.max(gt) - np.min(gt))
    return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(reco, gt):
    from skimage.metrics import structural_similarity as ssim
    data_range = (np.max(gt) - np.min(gt))
    return ssim(reco, gt, data_range=data_range)
    
seed=2022
whiten = 1
# define the inverse problem
data_args={'data_set':'simulation'}
# spat_args={'basis_opt':'Fourier','l':.1,'s':1.0,'q':1.0,'L':2000}
# temp_args={'ker_opt':'matern','l':.2,'q':1.0,'L':100}
# store_eig = True
# stpo = STEMPO(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)
msft = misfit(**data_args)
if msft.data_set=='simulation':
    truth = np.rot90(msft.truth,k=3,axes=(0,1))
else:
    truth = None


# models
pri_mdls=('q1p1','q2p2','iidT')
mdl_names=['STBP','STGP','time-independent']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './reconstruction'
if not os.path.exists(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl')):
    maps=[[]]*num_mdls
    funs=[[]]*num_mdls
    errs=[[]]*num_mdls
if os.path.exists(os.path.join(folder,'map_rle'+('_whiten' if whiten else '')+'.pckl')):
    f=open(os.path.join(folder,'map_rle'+('_whiten' if whiten else '')+'.pckl'),'rb')
    truth,rle_m,rle_s,loglik_m,loglik_s,psnr_m,psnr_s,ssim_m,ssim_s,haarpsi_m,haarpsi_s=pickle.load(f)
    f.close()
    print('map_rle'+('_whiten' if whiten else '')+'.pckl has been read!')
else:
    # store results
    rle_m=np.zeros(num_mdls)
    rle_s=np.zeros(num_mdls)
    loglik_m=np.zeros(num_mdls)
    loglik_s=np.zeros(num_mdls)
    psnr_m=np.zeros(num_mdls); psnr_s=np.zeros(num_mdls)
    ssim_m=np.zeros(num_mdls); ssim_s=np.zeros(num_mdls)
    haarpsi_m=np.zeros(num_mdls); haarpsi_s=np.zeros(num_mdls)
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        rle=[]; loglik=[]; psnr=[]; ssim=[]; haarpsi=[]
        num_read=0
        fld_m = os.path.join(folder,'MAP_Fourier_matern_'+('whiten_' if whiten else '')+pri_mdls[m])
        pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
        for f_i in pckl_files:
            try:
                f=open(os.path.join(fld_m,f_i),'rb')
                f_read=pickle.load(f)
                map=f_read[-3]
                rle.append(np.linalg.norm(map-truth)/np.linalg.norm(truth))
                loglik.append(-msft.cost(map))
                psnr.append(PSNR(map, truth))
                ssim.append(SSIM(map, truth))
                haarpsi_,_,_ = haar_psi_numpy(map,truth)
                haarpsi.append(haarpsi_)
                num_read+=1
                f.close()
                print(f_i+' has been read!')
            except:
                pass
        print('%d experiment(s) have been processed for %s prior model.' % (num_read, pri_mdls[m]))
        if num_read>0:
            rle_m[m] = np.median(rle)
            rle_s[m] = np.std(rle)
            loglik_m[m] = np.median(loglik)
            loglik_s[m] = np.std(loglik)
            psnr_m[m] = np.median(psnr)
            psnr_s[m] = np.std(psnr)
            ssim_m[m] = np.median(ssim)
            ssim_s[m] = np.std(ssim)
            haarpsi_m[m] = np.median(haarpsi)
            haarpsi_s[m] = np.std(haarpsi)
            # get the best for plotting
            if not os.path.exists(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl')):
                f_i=pckl_files[np.argmin(rle)]
                f=open(os.path.join(fld_m,f_i),'rb')
                f_read=pickle.load(f)
                map_f=f_read[-3] #(560, 560, 20), truth (560, 20, 560) plot_map_estimate!!!!!!!!!!!!!!!!!!!!!!!!!!!???????
                maps[m]=map_f
                funs[m]=np.pad(f_read[-2],(0,1000-len(f_read[-2])),mode='constant',constant_values=np.nan)
                errs[m]=np.pad(f_read[-1],(0,1000-len(f_read[-1])),mode='constant',constant_values=np.nan)
                f.close()
                print(f_i+' has been selected to print!')
    # save
    f=open(os.path.join(folder,'map_rle'+('_whiten' if whiten else '')+'.pckl'),'wb')
    pickle.dump([truth,rle_m,rle_s,loglik_m,loglik_s,psnr_m,psnr_s,ssim_m,ssim_s,haarpsi_m,haarpsi_s],f)
    f.close()
if not os.path.exists(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl')):
    maps=np.stack(maps)
    funs=np.stack(funs)
    errs=np.stack(errs)
    # save
    f=open(os.path.join(folder,'map_summary'+('_whiten' if whiten else '')+'.pckl'),'wb')
    pickle.dump([truth,maps,funs,errs],f)
    f.close()

# save
import pandas as pd
means = pd.DataFrame(data=np.vstack((rle_m,loglik_m,psnr_m,ssim_m,haarpsi_m)),columns=mdl_names[:num_mdls],index=['rle','log-lik','psnr','ssim','haarpsi'])
stds = pd.DataFrame(data=np.vstack((rle_s,loglik_s,psnr_s,ssim_s,haarpsi_s)),columns=mdl_names[:num_mdls],index=['rle','log-lik','psnr','ssim','haarpsi'])
means.to_csv(os.path.join(folder,'map_rle_means'+('_whiten' if whiten else '')+'.csv'),columns=mdl_names[:num_mdls])
stds.to_csv(os.path.join(folder,'map_rle_stds'+('_whiten' if whiten else '')+'.csv'),columns=mdl_names[:num_mdls])