## loo.py
## The implementation of Leave-one-out for data valuation

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def loo(trnX, trnY, devX, devY, clf):
    '''
    trnX, trnY - inputs/outputs of training examples
    
    '''
    N = trnX.shape[0]
    val, t = np.zeros((N)), 0
    for i in tqdm(range(N)): 
        # Shuffle the data
        # print(trnX.shape)
        acc_in, acc_ex = None, None
        Idx = list(range(N))
        # Include the data point i
        try:
            clf.fit(trnX, trnY)
            acc_in = accuracy_score(devY, clf.predict(devX))
        except ValueError:
            # Training set only has a single calss
            acc_in = accuracy_score(devY, [trnY[0]]*len(devY))
        # Exclude the data point i
        Idx.remove(i)
        tempX, tempY = trnX[Idx, :], trnY[Idx]
        # print(tempX.shape)
        try:
            clf.fit(tempX, tempY)
            acc_ex = accuracy_score(devY, clf.predict(devX))
        except ValueError:
            acc_ex = accuracy_score(devY, [trnY[0]]*len(devY))
        # print("acc_in = {}, acc_ex = {}".format(acc_in, acc_ex))
        val[i] = acc_in - acc_ex
    # print('val = {}'.format(val))
    return val
