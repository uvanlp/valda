import numpy as np
from tqdm import tqdm


## Local module 
from .pyclassifier import *


def inf_func(trnX, trnY, devX, devY, clf):
    '''
    trnX, trnY - 
    devX, devY - 
    clf - a classifier instance of PytorchClassifier
    '''

    trn_batch_size = clf.trn_batch_size
    dev_batch_size = clf.dev_batch_size
    epochs = clf.epochs

    if epochs > 0:
        print("Training models for IF with {} iterations ...".format(epochs))
        clf.fit(trnX, trnY)

    print("The current version of IF uses only first-order gradient, we will add the implementation of H^{-1} soon")
    print("In this implementation, large values mean important examples")


    trn_grads = clf.grad(trnX, trnY, batch_size=1)
    dev_grads = clf.grad(devX, devY, batch_size=dev_batch_size)


    infs = []
    for dev_grad in dev_grads:
        dev_grad = dev_grad[0]
        inf_up_loss = []
        for trn_grad in trn_grads:
            trn_grad = trn_grad[0]
            inf = 0
            for trn_grad_p, dev_grad_p in zip(trn_grad, dev_grad):
                assert trn_grad_p.size() == dev_grad_p.size()
                inf += torch.sum(trn_grad_p * dev_grad_p)
            inf_up_loss.append(inf.cpu().detach().numpy())
        infs.append(inf_up_loss)
    
    vals = list(np.sum(infs, axis=0))
    return vals

