## metrics.py
## Date: 01/26/2023
## Evaluation metrics used for data valuation

import numpy as np
import copy

def weighted_acc_drop(accs):
    ''' Weighted accuracy drop, please refer to (Schoch et al., 2022)
        for definition
    '''
    # accs = copy.copy(accs)
    accs.append(0.)
    accs = np.array(accs)
    diff = accs[:-1] - accs[1:]
    c_sum = np.cumsum(diff)
    weights = np.array(list(range(1, diff.shape[0]+1)))
    weights = 1.0/weights
    score = weights * c_sum
    return score.sum()


def pr_curve(target_list, ranked_list):
    ''' Compute P/R for two given lists
    '''
    p, r = [], []
    for idx in range(3, len(ranked_list)+1):
        partial_list = ranked_list[:idx]
        union = list(set(target_list) & set(partial_list))
        r.append(1.0*len(union)/len(target_list))
        p.append(1.0*len(union)/len(partial_list))
    score = auc(r, p)
    return (p, r, score)
