"""
Main function to obtain maximum a posterior (MAP) for the dynamic linear inverse problem of emoji.
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
from scipy import optimize
import timeit,time

# the inverse problem
from emoji import emoji

# basic settings
np.set_printoptions(precision=3, suppress=True)
import warnings
warnings.filterwarnings(action="once")

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('bas_NO', nargs='?', type=int, default=0)
    parser.add_argument('wav_NO', nargs='?', type=int, default=0)
    parser.add_argument('ker_NO', nargs='?', type=int, default=1)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('whiten', nargs='?', type=int, default=0) # choose to optimize in white noise representation space
    parser.add_argument('NCG', nargs='?', type=int, default=0) # choose to optimize with Newton conjugate gradient method
    parser.add_argument('bass', nargs='?', type=str, default=('Fourier','wavelet'))
    parser.add_argument('wavs', nargs='?', type=str, default=('Harr','Shannon','Meyer','MexHat','Poisson'))
    parser.add_argument('kers', nargs='?', type=str, default=('powexp','matern'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(seed)
    
    # define emoji Bayesian inverse problem
    data_args={'data_set':'60proj','data_thinning':2}
    spat_args={'basis_opt':args.bass[args.bas_NO],'l':1,'s':1,'q':args.q,'L':2000}
    if spat_args['basis_opt']=='wavelet': spat_args['wvlet_typ']=args.wavs[args.wav_NO]
    temp_args={'ker_opt':args.kers[args.ker_NO],'l':.5,'q':1.0,'L':100}
    store_eig = True
    emj = emoji(**data_args, spat_args=spat_args, temp_args=temp_args, store_eig=store_eig, seed=seed)#, init_param=True)
    # truth = emj.misfit.truth # no truth
    
    # optimize to get MAP
    print("Obtaining MAP estimate for %s spatial basis %s with %s kernel %s ..." % (args.bass[args.bas_NO], '('+args.wavs[args.wav_NO]+')' if spat_args['basis_opt']=='wavelet' else '', args.kers[args.ker_NO], {True:'using Newton CG',False:''}[args.NCG]))
    
    if not hasattr(emj,'init_parameter'): emj._init_param()#init_opt='LSE',lmda=10)
    param0 = emj.init_parameter
    if emj.prior.space=='fun': param0=emj.prior.vec2fun(param0)
    if args.whiten: param0 = emj.whiten.stbp2wn(param0).flatten(order='F')
    fun = lambda parameter: emj._get_misfit(emj.whiten.wn2stbp(parameter) if args.whiten else parameter, MF_only=False)
    def grad(parameter):
        param = emj.whiten.wn2stbp(parameter) if args.whiten else parameter
        g = emj._get_grad(param, MF_only=False)
        if args.whiten: g = emj.whiten.wn2stbp(parameter, 1)(g, adj=True)
        return g.squeeze()
    def hessp(parameter,v):
        param = emj.whiten.wn2stbp(parameter) if args.whiten else parameter
        Hv = emj._get_HessApply(param, MF_only=False)(emj.whiten.wn2stbp(parameter,1)(v) if args.whiten else v)
        if args.whiten:
            Hv = emj.whiten.wn2stbp(parameter, 1)(Hv, adj=True) 
            Hv+= emj.whiten.wn2stbp(parameter, 2)(v, emj._get_grad(param, MF_only=False), adj=True)
        return Hv.squeeze()
    h=1e-6; v=np.random.randn(emj.prior.bsv.L*emj.prior.qep.N) if args.whiten else emj.prior.sample()
    # if args.whiten: v=emj.whiten.stbp2wn(v).flatten(order='F')
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
        # Xi_=emj.whiten.wn2stbp(Xi) if args.whiten else Xi
        # ERR.append(np.linalg.norm((emj.prior.vec2fun(Xi_) if emj.prior.space=='vec' else Xi_) -truth.flatten(order='F'))/np.linalg.norm(truth))
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
    # solve for MAP
    start = time.time()
    if args.NCG:
        # res = optimize.minimize(fun, param0, method='Newton-CG', jac=grad, hessp=hessp, callback=call_back, options={'maxiter':100,'disp':True})
        res = optimize.minimize(fun, param0, method='trust-ncg', jac=grad, hessp=hessp, callback=call_back, options={'maxiter':1000,'disp':True})
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
    map_v=emj.whiten.wn2stbp(res.x) if args.whiten else res.x
    map_f=emj.prior.vec2fun(map_v) if emj.prior.space=='vec' else map_v; funs=np.stack(FUN); errs=[] if len(ERR)==0 else np.stack(ERR)
    map_f=map_f.reshape(np.append(emj.misfit.sz_x,emj.misfit.sz_t),order='F').swapaxes(0,1)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    f_name=spat_args['basis_opt']+('_'+spat_args['wvlet_typ'] if spat_args['basis_opt']=='wavelet' else '')+'_'+temp_args['ker_opt']+('_whiten' if args.whiten else '')+('_NCG' if args.NCG else '')
    savepath=os.path.join(os.getcwd(),'./reconstruction/MAP_'+f_name)
    if not os.path.exists(savepath): os.makedirs(savepath)
    # save
    filename='emj_MAP_dim'+str(len(map_f))+'_'+f_name+'_'+ctime+'.pckl'
    f=open(os.path.join(savepath,filename),'wb')
    # pickle.dump([truth, map_f, funs, errs],f)
    pickle.dump([spat_args, temp_args, map_f, funs, errs],f)
    f.close()
    # plot
    emj.misfit.plot_reconstruction(rcstr_imgs=map_f, save_imgs=True, save_path=savepath)

if __name__ == '__main__':
    main()
    # n_seed = 10; i=0; n_success=0
    # while n_success < n_seed:
    #     seed_i=2022+i*10
    #     try:
    #         print("Running for seed %d ...\n"% (seed_i))
    #         main(seed=seed_i)
    #         n_success+=1
    #     except Exception as e:
    #         print(e)
    #         pass
    #     i+=1