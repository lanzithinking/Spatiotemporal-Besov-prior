"Plot comparison of various models"

import os
import numpy as np

os.makedirs('./results', exist_ok=True)

qs=[0, 1, 2]
algs=['q'+str(i) for i in qs]
algs[0] = 'iidT'
num_algs=len(algs)

# set the format
folder = './results'
header = ('Method', 'RLE', 'NLP', 'time')
incl_header = True
fmt = ('%s',)+('%.4f',)*3
# obtain results
formats = ('i4','S20')+('f4',)*3
f_read=np.loadtxt(os.path.join('./results','noaatmp_MAP_dim241920_hldt0.1.txt'), delimiter=',', dtype={'names':('seed',)+header,'formats':formats})

for j in range(num_algs):
    stats = []
    for r_i in f_read:
        if algs[j] in r_i[1].astype(str): stats.append(list(r_i)[2:])
    stats = np.stack(stats)
    stats = stats[np.isfinite(stats).all(axis=1)]
    with open(os.path.join(folder, 'summary_mean.csv'),'ab') as f:
        np.savetxt(f, np.concatenate([[algs[j]],stats.mean(0)])[None,:], fmt='%s', delimiter=',', header=','.join(header) if incl_header else '')
    f.close()
    with open(os.path.join(folder, 'summary_std.csv'),'ab') as f:
        np.savetxt(f, np.concatenate([[algs[j]],stats.std(0,ddof=1)])[None,:], fmt='%s', delimiter=',', header=','.join(header) if incl_header else '')
    f.close()
    incl_header = False
