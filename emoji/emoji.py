"""
Utility functions for the emoji dataset.
"""

import requests
from scipy import sparse
import numpy as np
import h5py
from gks_tools import *
from os.path import exists

def get_emoji_data():

    if exists('./data/DataDynamic_128x30.mat'):
        print('data already downloaded.')

    else:
        print("downloading...")
        r = requests.get('https://zenodo.org/record/1183532/files/DataDynamic_128x30.mat')

        with open('./data/DataDynamic_128x30.mat', "wb") as file:

            file.write(r.content)

        print("downloaded.")



def generate_emoji(noise_level):

    get_emoji_data()

    with h5py.File('./data/DataDynamic_128x30.mat', 'r') as f:
        A = sparse.csc_matrix((f["A"]["data"], f["A"]["ir"], f["A"]["jc"]))
        normA = np.array(f['normA'])
        sinogram = np.array(f['sinogram']).T

    T = 33
    N = np.sqrt(A.shape[1] / T)
    [mm, nn] = sinogram.shape

    ind = []

    for ii in range(int(nn /3)):

        ind.extend( np.arange(0,mm) + (3*ii)*mm )

    m2 = sinogram[:, 0::3]

    A_small = A[ind, :]

    b = m2
    nt = int(T)
    nx = int(N)
    ny = int(N)
    b = b.reshape(-1, 1, order='F').squeeze()

    AA = list(range(T))
    B = list(range(T))

    delta = 0 # no added noise for this dataset

    for ii in range(T):

        AA[ii] = A_small[ 2170*(ii):2170*(ii+1), 16384*ii:16384*(ii+1) ]
        B[ii] = b[ 2170*(ii) : 2170*(ii+1) ]

    return (A_small, b, AA, B, nx, ny, nt, 0)


def first_derivative_operator(n):

    D = sparse.spdiags( data=np.ones(n-1) , diags=-1, m=n, n=n)
    L = sparse.identity(n)-D

    return L



def first_derivative_operator_2d(nx, ny):

    D_x = first_derivative_operator(nx)
    D_y = first_derivative_operator(ny)

    IDx = sparse.kron( sparse.identity(nx), D_x)
    DyI = sparse.kron(D_y, sparse.identity(ny))

    D_spatial = sparse.vstack((IDx, DyI))

    return D_spatial

def anisoTV_operator(nx, ny, nt):

    D_spatial = first_derivative_operator_2d(nx,ny)

    D_time = first_derivative_operator(nt)[:-1, :]

    ID_spatial = sparse.kron( sparse.identity(nt), D_spatial)

    D_timeI = sparse.kron(D_time, sparse.identity(nx**2))

    L = sparse.vstack((ID_spatial, D_timeI))

    return L


if __name__ == "__main__":

    (A, b, AA, B, nx, ny, nt, delta) = generate_emoji(0)

    L = anisoTV_operator(nx,ny,nt)
    xhat = GKS(A, b, L, 1, 5, 0, 0)

    xx = np.reshape(xhat, (128,128,33), order="F")
    plt.set_cmap('Greys')
    for i in range(xx.shape[2]):
        plt.imshow(xx[:,:,i])
        plt.savefig('./data/emoji_'+str(i)+'.png',bbox_inches='tight')
        # plt.show()