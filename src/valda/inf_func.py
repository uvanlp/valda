import numpy as np
from tqdm import tqdm


## Local module 
from .pyclassifier import *


def inf_func(trnX, trnY, devX, devY, clf,
             epochs=5, trn_batch_size=1, dev_batch_size=16):

    if epochs > 0:
        print("Training models for IF with {} iterations ...".format(epochs))
        clf.fit(trnX, trnY, epochs=epochs, batch_size=trn_batch_size)

    print("The current version of IF uses only first-order gradient, we will add the implementation of H^{-1} soon")
    print("In this implementation, large values mean important examples")
    train_grads = clf.grad(trnX, trnY, batch_size=1)
    test_grads = clf.grad(devX, devY, batch_size=dev_batch_size)

    infs = []
    for test_grad in test_grads:
        inf_up_loss = []
        for train_grad in train_grads:
            inf = 0
            for train_grad_p, test_grad_p in zip(train_grad, test_grad):
                inf += torch.sum(train_grad_p * test_grad_p)
            inf_up_loss.append(inf.cpu().detach().numpy())
        infs.append(inf_up_loss)
    
    vals = list(np.sum(infs, axis=0))
    return vals

