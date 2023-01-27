## beta_shapley.py
## Implementation of Beta Shapley

import numpy as np
from random import shuffle, seed, randint, sample, choice
from tqdm import tqdm
from sklearn.metrics import accuracy_score


## Local module 
from .util import *


def beta_shapley(trnX, trnY, devX, devY, clf, alpha=1.0,
                     beta=1.0, rho=1.0005, K=10, T=10):
    """
    alpha, beta - parameters for Beta distribution
    rho - GR statistic threshold
    K - number of Markov chains
    T - upper bound of iterations
    """
    N = trnX.shape[0]
    Idx = list(range(N)) # Indices
    val, t = np.zeros((N, K, T+1)), 0
    rho_hat = 2*rho
    # val_N_list = []

    # Data information
    N = len(trnY)
    # Computation
    # while np.any(rho_hat >= rho):
    for t in tqdm(range(1, T+1)):
        # print("Iteration: {}".format(t))
        for j in range(N):
            for k in range(K):
                Idx = list(range(N))
                Idx.remove(j) # remove j
                s = randint(1, N-1)
                sub_Idx = sample(Idx, s)
                acc_ex, acc_in = None, None
                # =========================
                trnX_ex, trnY_ex = trnX[sub_Idx, :], trnY[sub_Idx]
                try:
                    clf.fit(trnX_ex, trnY_ex)
                    acc_ex = accuracy_score(devY, clf.predict(devX))
                except ValueError:
                    acc_ex = accuracy_score(devY, [trnY_ex[0]]*len(devY))
                # =========================
                sub_Idx.append(j) # Add example j back for training
                trnX_in, trnY_in = trnX[sub_Idx, :], trnY[sub_Idx]
                try:
                    clf.fit(trnX_in, trnY_in)
                    acc_in = accuracy_score(devY, clf.predict(devX))
                except ValueError:
                    acc_in = accuracy_score(devY, [trnY_in[0]]*len(devY))
                # Update the value
                val[j,k,t] = ((t-1)*val[j,k,t-1])/t + (weight(j+1, N, alpha, beta)/t)*(acc_in - acc_ex)
        # Update the Gelman-Rubin statistic rho_hat
        if t > 3:
            rho_hat = gr_statistic(val, t) # A temp solution for stopping
            # print("rho_hat = {}".format(rho_hat[:5]))
        if np.all(rho_hat < rho):
            # terminate the outer loop earlier
            break
    # average all the sample values
    # val_mean = val[:,:,1:t+1].mean(axis=2).mean(axis=1) # N
    val_last = val[:,:,t].mean(axis=1)
    # print(val_last)
    return val_last
