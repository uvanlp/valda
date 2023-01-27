## data_removal.py
## Evaluate the performance of data valuation by removing one data point
## at a time from the training set

from sklearn.metrics import accuracy_score, auc
from sklearn.linear_model import LogisticRegression as LR

import operator

def data_removal(vals, trnX, trnY, tstX, tstY, clf=None,
                     remove_high_value=True):
    '''
    trnX, trnY - training examples
    tstX, tstY - test examples
    vals - a Python dict that contains data indices and values
    clf - the classifier that will be used for evaluation
    '''
    # Create data indices for data removal
    N = trnX.shape[0]
    Idx_keep = [True]*N

    if clf is None:
        clf = LR(solver="liblinear", max_iter=500, random_state=0)
    # Sorted the data indices with a descreasing order
    sorted_dct = sorted(vals.items(), key=operator.itemgetter(1), reverse=True)
    # Accuracy list
    accs = []
    if remove_high_value:
      lst = range(N)
    else:
      lst = range(N-1, -1, -1)
    # Compute 
    clf.fit(trnX, trnY)
    acc = accuracy_score(clf.predict(tstX), tstY)
    accs.append(acc)
    for k in lst: 
        # print(k)
        Idx_keep[sorted_dct[k][0]] = False
        trnX_k = trnX[Idx_keep, :]
        trnY_k = trnY[Idx_keep]
        try:
            clf.fit(trnX_k, trnY_k)
            # print('trnX_k.shape = {}'.format(trnX_k.shape))
            acc = accuracy_score(clf.predict(tstX), tstY)
            # print('acc = {}'.format(acc))
            accs.append(acc)
        except ValueError:
            # print("Training with data from a single class")
            accs.append(0.0)    
    return accs
