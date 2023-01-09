"""
This is to test the white noise representation of Q-EP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append( "../" )
from util.stbp.qEP import qEP

# set random seed
np.random.seed(2022)

# set up
dim = 4
q = 1
prior_args={'ker_opt':'matern','l':.5,'q':q,'L':100}
store_eig=True
# define Q-EP process
x = np.linspace(0,1,dim)
qep = qEP(x=x, store_eig=store_eig, **prior_args)

# white noise representation
nmlz = lambda z,q=1: z/np.linalg.norm(z,axis=1)[:,None]**q
Lmd = lambda z,q=qep.q: qep.act(nmlz(z.reshape((-1,qep.N),order='F'),1-2/q),alpha=0.5,transp=True)

# generate samples
n_samp = 5000
samp_qep = qep.rnd(n=n_samp).T # sample by Q-EP definition
samp_wnr = Lmd(np.random.randn(n_samp,dim)) # sample by white noise representation
samp2plot = pd.DataFrame( np.vstack(( np.hstack((samp_qep,np.tile(0,(n_samp,1)))), np.hstack((samp_wnr,np.tile(1,(n_samp,1)))) )),
                        columns=['$x\_{}$'.format(j) for j in range(dim)]+['source'])
# samp2plot.rename(columns={samp2plot.columns.values[-1]: 'source'},inplace=True)

# plot
g=sns.PairGrid(data=samp2plot,vars=samp2plot.columns[:-1], hue='source')
g.map_upper(sns.scatterplot, size=5)#,style=samp2plot.loc[:,"source"])
g.map_diag(sns.kdeplot)
g.map_lower(sns.kdeplot)
# g.add_legend(title='method')#,legend_data={0:'Q-EP',1:'wn-rep'})
plt.legend(labels=['Q-EP','wn-rep'])
plt.show()