from imghdr import what
from random import betavariate
from re import X
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

def lanczos_biortho_pasha(A, guess, iter):

    # dimensions
    N = len(guess)
    M = len(A.T @ guess)

    # preallocate
    U = np.zeros(shape=(N, iter+1))
    V = np.zeros(shape=(M, iter))

    v = np.zeros(shape=(M,1))
    # normalize initial guess
    beta = np.linalg.norm(guess)

    assert beta != 0

    u = guess/beta

    U[:,[0]] = u
     
    alphas = np.zeros(shape=(iter+1))
    betas = np.zeros(shape=(iter+2))
    # begin bidiagonalization

    for ii in range(0,iter):

        r = A.T @ u
        r = r - beta*v

        for jj in range(0,ii-1): # reorthogonalization

            r = r - (V[:,[jj]].T @ r) * V[:,[jj]]

        alpha = np.linalg.norm(r)
        alphas[ii] = alpha
        v = r/alpha


        V[:,ii] = v.flatten()

        p = A @ v

        p = p - alpha*u


        for jj in range(0, ii):

            p = p - (U[:,[jj]].T @ p) * U[:,[jj]]

        beta = np.linalg.norm(p)
        betas[ii] = beta
        u = p / beta

        U[:, [ii+1]] = u

    return (U, betas, alphas, V)


def GKS(A, b, L, lanczos_dim, iter, delta, eta):

    (U, beta, V) = lanczos_biortho_pasha(A, b, lanczos_dim) # Find a small basis V

    for ii in range(iter):

        (Q_A, R_A) = np.linalg.qr(A @ V) # Project A into V, separate into Q and R

        (Q_L, R_L) = np.linalg.qr(L @ V) # Project L into V, separate into Q and R

        lambdah = 0.0001 # set an arbitrary lambda

        bhat = (Q_A.T @ b).reshape(-1,1) # Project b

        R_stacked = np.vstack([R_A, lambdah*R_L]) # Stack projected operators


        b_stacked = np.vstack([bhat, np.zeros(shape=(R_L.shape[0], 1)) ]) # pad with zeros

        y, _,_,_ = np.linalg.lstsq(R_stacked, b_stacked) # get least squares solution

        x = V @ y # project y back
        r = b.reshape(-1,1) - (A @ x).reshape(-1,1) # get residual
        ra = A.T@r
        rb = L@x 
        rb = L.T@rb 
        r = ra + lambdah*rb
        r = r - V@(V.T@r)
        r = r - V@(V.T@r)
        normed_r = r / np.linalg.norm(r) # normalize residual
        V = np.hstack([V, normed_r]) # add residual to basis
        # V, _ = np.linalg.qr(V) # orthonormalize basis using QR
        print(V.shape)
    return x


if __name__ == "__main__":

    A = np.random.rand(10,10)
    x = np.random.rand(10)

    """(U, beta, V) = lanczos_biortho_pasha(A, b, 5)

    print(U.shape)
    print(beta)
    print(V.shape)

    print(np.linalg.norm(A @ V - W @ T, 'fro'))
    print(np.round(U.T @ A @ V, 2))"""

    # b = A @ x

    # xhat = GKS(A, b, np.eye(10), 3, 5, 0, 0)

    # print(np.linalg.norm(x - xhat)/np.linalg.norm(x))