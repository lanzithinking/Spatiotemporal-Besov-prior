"""
Main function to obtain maximum a posterior (MAP) for the example of NOAA temperature.
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
from scipy import optimize
import timeit,time

# the inverse problem
from NOAATMP import NOAATMP

# basic settings
np.set_printoptions(precision=3, suppress=True)
import warnings
warnings.filterwarnings(action="once")

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('bas_NO', nargs='?', type=int, default=0)
    parser.add_argument('wav_NO', nargs='?', type=int, default=0)
    parser.add_argument('ker_NO', nargs='?', type=int, default=1)
    parser.add_argument('q', nargs='?', type=float, default=2)
    parser.add_argument('whiten', nargs='?', type=int, default=1) # choose to optimize in white noise representation space
    parser.add_argument('NCG', nargs='?', type=int, default=0) # choose to optimize with Newton conjugate gradient method
    parser.add_argument('bass', nargs='?', type=str, default=('Fourier','wavelet'))
    parser.add_argument('wavs', nargs='?', type=str, default=('Harr','Shannon','Meyer','MexHat','Poisson'))
    parser.add_argument('kers', nargs='?', type=str, default=('powexp','matern'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(seed)
    
    # define Bayesian inverse problem
    if args.q==0:
        temp_corl=1e-10
        args.q=1
    else:
        temp_corl=0.2
    data_args={'impute_missing':False,'nzlvl':.1,'holdout_percent':0.1,'random_seed':seed}
    spat_args={'basis_opt':args.bass[args.bas_NO],'sigma2':100,'l':.1,'s':1,'q':args.q,'L':2000}
    if spat_args['basis_opt']=='wavelet': spat_args['wvlet_typ']=args.wavs[args.wav_NO]
    temp_args={'ker_opt':args.kers[args.ker_NO],'sigma2':10, 'l':temp_corl,'s':1,'q':args.q,'L':100}
    store_eig = True
    noaatmp = NOAATMP(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
    # if hasattr(noaatmp.misfit, 'truth'):
    #     truth = noaatmp.misfit.truth
    if noaatmp.misfit.hldt_pcnt>0:
        truth = noaatmp.misfit.obs[noaatmp.misfit.holdout_msk]
    else:
        truth = noaatmp.misfit._impute(axis=[0,1])
    noaatmp.misfit.truth = truth
    
    # optimize to get MAP
    print("Obtaining MAP estimate for %s spatial basis %s with %s kernel %s ..." % (args.bass[args.bas_NO], '('+args.wavs[args.wav_NO]+')' if spat_args['basis_opt']=='wavelet' else '', args.kers[args.ker_NO], {True:'using Newton CG',False:''}[args.NCG]))
    
    if not hasattr(noaatmp,'init_parameter'): noaatmp._init_param(init_opt='LSE', axis=[0,1])
    param0 = noaatmp.init_parameter
    if noaatmp.prior.space=='fun': param0=noaatmp.prior.vec2fun(param0)
    # param0 = noaatmp.prior.sample()
    if args.whiten: param0 = noaatmp.whiten.stbp2wn(param0).flatten(order='F')
    fun = lambda parameter: noaatmp._get_misfit(noaatmp.whiten.wn2stbp(parameter) if args.whiten else parameter, MF_only=False)
    def grad(parameter):
        param = noaatmp.whiten.wn2stbp(parameter) if args.whiten else parameter
        g = noaatmp._get_grad(param, MF_only=False)
        if args.whiten: g = noaatmp.whiten.wn2stbp(parameter, 1)(g, adj=True)
        return g.squeeze()
    def hessp(parameter,v):
        param = noaatmp.whiten.wn2stbp(parameter) if args.whiten else parameter
        Hv = noaatmp._get_HessApply(param, MF_only=False)(noaatmp.whiten.wn2stbp(parameter,1)(v) if args.whiten else v)
        if args.whiten:
            Hv = noaatmp.whiten.wn2stbp(parameter, 1)(Hv, adj=True) 
            Hv+= noaatmp.whiten.wn2stbp(parameter, 2)(v, noaatmp._get_grad(param, MF_only=False), adj=True)
        return Hv.squeeze()
    h=1e-7; v=noaatmp.whiten.sample() if args.whiten else noaatmp.prior.sample()
    # if args.whiten: v=noaatmp.whiten.stbp2wn(v).flatten(order='F')
    f,g,Hv=fun(param0),grad(param0),hessp(param0,v)
    f1,g1=fun(param0+h*v),grad(param0+h*v)
    print('error in gradient: %0.8f' %(abs((f1-f)/h-g.dot(v))/np.linalg.norm(v)))
    print('error in Hessian: %0.8f' %(np.linalg.norm((g1-g)/h-Hv)/np.linalg.norm(v)))
    
    global Nfeval,FUN,ERR
    Nfeval=1; FUN=[]; ERR=[];
    def call_back(Xi):
        global Nfeval,FUN,ERR
        fval=fun(Xi)
        print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], fval))
        Nfeval += 1
        FUN.append(fval)
        if hasattr(noaatmp.misfit, 'truth'):
            Xi_=noaatmp.whiten.wn2stbp(Xi) if args.whiten else Xi
            U_= (noaatmp.prior.vec2fun(Xi_) if noaatmp.prior.space=='vec' else Xi_).reshape(noaatmp.misfit.obs.shape, order='F')
            if noaatmp.misfit.hldt_pcnt>0: U_=U_[noaatmp.misfit.holdout_msk]
            ERR.append(np.linalg.norm(U_ -truth)/np.linalg.norm(truth))
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
    # solve for MAP
    start = time.time()
    if args.NCG:
        # res = optimize.minimize(fun, param0, method='Newton-CG', jac=grad, hessp=hessp, callback=call_back, options={'maxiter':100,'disp':True})
        res = optimize.minimize(fun, param0, method='trust-ncg', jac=grad, hessp=hessp, callback=call_back, options={'maxiter':100,'disp':True})
    else:
        res = optimize.minimize(fun, param0, method='L-BFGS-B', jac=grad, callback=call_back, options={'maxiter':1000,'disp':True})
    end = time.time()
    print('\nTime used is %.4f' % (end-start))
    # print out info
    if res.success:
        print('\nConverged in ', res.nit, ' iterations.')
    else:
        print('\nNot Converged.')
    print('Final function value: %.4f.\n' % res.fun)
    
    # store the results
    map_v=noaatmp.whiten.wn2stbp(res.x) if args.whiten else res.x
    map_f=(noaatmp.prior.vec2fun(map_v) if noaatmp.prior.space=='vec' else map_v).reshape(noaatmp.misfit.obs.shape, order='F')
    if hasattr(noaatmp.misfit, 'truth'):
        #  compare it with the truth
        map2cmp = map_f[noaatmp.misfit.holdout_msk] if noaatmp.misfit.hldt_pcnt>0 else map_f
        relerr = np.linalg.norm(map2cmp -truth)/np.linalg.norm(truth)
        print('Relative error of MAP compared with the truth %.2f%%' % (relerr*100))
    map_f=map_f.reshape(np.append(noaatmp.misfit.sz_x,noaatmp.misfit.sz_t),order='F')
    funs=np.stack(FUN); errs=[] if len(ERR)==0 else np.stack(ERR)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    f_name=spat_args['basis_opt']+('_'+spat_args['wvlet_typ'] if spat_args['basis_opt']=='wavelet' else '')+'_'+temp_args['ker_opt']\
           +('_whiten' if args.whiten else '')+('_NCG' if args.NCG else '')+('_iidT' if temp_args['l']<1e-5 else '_q'+str(args.q))+'_hldt'+str(noaatmp.misfit.hldt_pcnt)
    savepath=os.path.join(os.getcwd(),'./reconstruction/noaatmp_I{}_J{}_'.format(noaatmp.prior.I, noaatmp.prior.J)+'/MAP_'+f_name)
    if not os.path.exists(savepath): os.makedirs(savepath)
    # # save
    # filename='noaatmp_MAP_dim'+str(map_f.size)+'_'+f_name+'_'+ctime+'.pckl'
    # f=open(os.path.join(savepath,filename),'wb')
    # # pickle.dump([truth, map_f, funs, errs],f)
    # pickle.dump([spat_args, temp_args, map_f, funs, errs],f)
    # f.close()
    # # plot
    # noaatmp.misfit.plot_reconstruction(rcstr_imgs=map_f, save_imgs=True, save_path=savepath, snap_idx=np.linspace(0,noaatmp.misfit.sz_t,num=10,endpoint=False).astype(int), label=False)
    
    # save to text
    os.makedirs('./results', exist_ok=True)
    stats = np.array([relerr, funs[-1], end-start])
    stats = np.array([seed,('_iidT' if temp_args['l']<1e-5 else '_q'+str(args.q))]+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'RLE', 'NLP', 'time']
    with open(os.path.join('./results','noaatmp_MAP_dim'+str(map_f.size)+'_hldt'+str(noaatmp.misfit.hldt_pcnt)+'.txt'),'ab') as f:
        np.savetxt(f,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2022 else '')

if __name__ == '__main__':
    # main()
    n_seed = 10; i=0; n_success=0; n_failure=0
    while n_success < n_seed and n_failure < 10* n_seed:
        seed_i=2022+i*10
        try:
            print("Running for seed %d ...\n"% (seed_i))
            main(seed=seed_i)
            n_success+=1
        except Exception as e:
            print(e)
            n_failure+=1
            pass
        i+=1