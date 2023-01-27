import numpy as np
from math import factorial
from sklearn.metrics import accuracy_score, auc
import copy


def weight(j, n, alpha=1.0, beta=1.0):
    log_1, log_2, log_3 = 0.0, 0.0, 0.0
    for k in range(1, j):
        log_1 += np.log(beta + k - 1)
    for k in range(1, n-j+1):
        log_2 += np.log(alpha + k - 1)
    for k in range(1, n):
        log_3 += np.log(alpha + beta + k - 1)
    log_total = np.log(n) + log_1 + log_2 - log_3
    # print("n = {}, j = {}".format(n, j))
    log_comb = None
    if n <= 20:
        log_comb = np.log(factorial(n-1))
    else:
        log_comb = (n-1)*(np.log(n-1) - 1)
    if j <= 20:
        log_comb -= np.log(factorial(j-1))
    else:
        log_comb -= (j-1)*(np.log(j-1) - 1)
    if (n-j) <= 20:
        log_comb -= np.log(factorial(n-j))
    else:
        log_comb -= (n-j)*(np.log(n-j) - 1)
    # print("log_total = {}, log_comb = {}".format(log_total, log_comb))
    v = np.exp(log_comb + log_total)
    # print("v = {}".format(v))
    return v


def gr_statistic(val,t):
    v = val[:,:,1:t+1]
    sample_var = np.var(v, axis=2, ddof=1) # N x K, along dimension T
    mean_sample_var = np.mean(sample_var, axis=1) # N, along dimension K, s^2 in the paper
    sample_mean = np.mean(v, axis=2) # N x K, along dimension T
    sample_mean_var = np.var(sample_mean, axis=1, ddof=1) # N, along dimension K, B/n in the paper
    sigma_hat_2 = ((t-1)*mean_sample_var)/t + sample_mean_var
    rho_hat = np.sqrt(sigma_hat_2/(mean_sample_var + 1e-4))
    return rho_hat






def data_removal_figure(neg_lab, pos_lab, trnX, trnY, devX, devY, sorted_dct, clf_label, remove_high_value=True):
    # Create data indices for data removal
    N = trnX.shape[0]
    Idx_keep = [True]*N
    # Accuracy list
    accs = []
    if remove_high_value:
      lst = range(N)
    else:
      lst = range(N-1, -1, -1)
    # Compute
    clf = Classifier(clf_label)
    clf.fit(trnX, trnY)
    dev = zip(devX, devY)
    dev = list(dev)
    dev_X0 = []
    dev_X1 = []
    dev_Y0 = []
    dev_Y1 = []

    for i in dev:
        if i[1] == pos_lab:
            dev_X1.append(i[0])
            dev_Y1.append(i[1])
        elif i[1] == neg_lab:
            dev_X0.append(i[0])
            dev_Y0.append(i[1])
        else:
            print(i)


    acc_0 = accuracy_score(dev_Y0, clf.predict(dev_X0), normalize=False)/len(devY)
    acc_1 = accuracy_score(dev_Y1, clf.predict(dev_X1), normalize=False)/len(devY)
    print(acc_0, acc_1)

    accs_0 = []
    accs_1 = []
    acc = accuracy_score(clf.predict(devX), devY)#/len(devY)
    accs.append(acc)
    accs_0.append(acc_0)
    accs_1.append(acc_1)
    vals = []
    labels = []
    points = []
    ks = []
    for k in lst:
        # print(k)
        Idx_keep = [True] * N
        Idx_keep[sorted_dct[k][0]] = False
        trnX_k = trnX[Idx_keep, :]
        trnY_k = trnY[Idx_keep]
        clf = Classifier(clf_label)
        try:
            clf.fit(trnX_k, trnY_k)
            # print('trnX_k.shape = {}'.format(trnX_k.shape))
            labels.append(trnY[k])
            points.append(trnX[k])
            acc = accuracy_score(clf.predict(devX), devY)
            acc_0 = accuracy_score(dev_Y0, clf.predict(dev_X0), normalize=False)/len(devY)
            acc_1 = accuracy_score(dev_Y1, clf.predict(dev_X1), normalize=False)/len(devY)
            # print('acc = {}'.format(acc))
            ks.append(k)
            accs.append(acc)
            accs_0.append(acc_0)
            accs_1.append(acc_1)
            vals.append(sorted_dct[k][1])
        except ValueError:
            # print("Training with data from a single class")
            accs.append(0.0)
    return accs, accs_0, accs_1, vals, labels, points, ks







        
