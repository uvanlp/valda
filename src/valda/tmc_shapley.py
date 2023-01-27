import numpy as np
from random import shuffle, seed, randint, sample, choice
from tqdm import tqdm
from sklearn.metrics import accuracy_score



def tmc_shapley(trnX, trnY, devX, devY, clf, T=20, epsilon=0.001):
    N = trnX.shape[0]
    Idx = list(range(N)) # Indices
    val, t = np.zeros((N)), 0
    # Start calculation
    val_T = np.zeros((T, N))
    for t in tqdm(range(1, T+1)):
        # Shuffle the data
        shuffle(Idx)
        val_t = np.zeros((N+1))
        # pre-computed values (with all training data/without training data)
        try:
          clf.fit(trnX, trnY)
          val_t[N] = accuracy_score(devY, clf.predict(devX))
        except ValueError:
          # Training set only has a single calss
          val_t[N] = accuracy_score(devY, [trnY[0]]*len(devY))
        #
        for j in range(1,N+1):
            if abs(val_t[N] - val_t[j-1]) < epsilon:
                val_t[j] = val_t[j-1]
            else:
                # Extract the first $j$ data points
                trnX_j = trnX[Idx[:j],:]
                # print("trnX_j.shape = {}".format(trnX_j.shape))
                trnY_j = trnY[Idx[:j]]
                try:
                    clf.fit(trnX_j, trnY_j)
                    val_t[j] = accuracy_score(devY, clf.predict(devX))
                except ValueError:
                    # the majority vote
                    val_t[j] = accuracy_score(devY, [trnY_j[0]]*len(devY))
        # Update the data shapley values
        val[Idx] = ((1.0*(t-1)/t))*val[Idx] + (1.0/t)*(val_t[1:] - val_t[:N])
        val_T[t-1,:] = (np.array(val_t)[1:])[Idx]
    return val
