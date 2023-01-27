import numpy as np
from tqdm import tqdm


## Local module 
from .pyclassifier import *

"""
Important Params:
ys: scalar to be differentiated
params: list of vectors (torch.tensors) w.r.t. each of which the hessian is computed
vs: the list of vectors each of which is to be multiplied to the hessian w.r.t. each parameter
params2: another list of params for second `grad` call in case the second derivation is w.r.t. a different set of parameters
"""


def hessian_vector_product(ys, params, vs, params2=None):
    grads1 = grad(ys, params, create_graph=True)
    if params2 is not None:
        params = params2

    grads2 = grad(grads1, params, grad_outputs=vs)
    return grads2


# Each output in the list is obtained by differentiating `ys` w.r.t. only a single parameter.
# Returns: a list of hessians of `ys` w.r.t. each parameter in `params`, i.e. differentiate `ys` twice w.r.t. each parameter. One hessian per param.
"""
Important Params:
ys: scalar that is to be differentiated
params: list of torch.tensors, hessian is computed for each
"""


def hessians(ys, params):
    jacobians = grad(ys, params, create_graph=True)

    outputs = []  # container for hessians
    for j, param in zip(jacobians, params):
        hess = []
        j_flat = j.flatten()
        for i in range(len(j_flat)):
            grad_outputs = torch.zeros_like(j_flat)
            grad_outputs[i] = 1
            grad2 = grad(j_flat, param, grad_outputs=grad_outputs, retain_graph=True)[0]
            hess.append(grad2)
        outputs.append(torch.stack(hess).reshape(j.shape + param.shape))
    return outputs


# Compute product of inverse hessian of empirical risk and given vector 'v', computed numerically using LiSSA.
# Return a list of inverse-hvps, computed for each param.

'''
Important params:
vs: list of vectors in the inverse-hvp, one per parameter
batch_size: size of minibatch sample at each iteration
scale: the factor to scale down loss (to keep hessian <= I)
damping: lambda added to guarantee hessian be p.d.
num_repeats: hyperparameter 'r' in in the paper (to reduce variance)
recursion_depth: number of iterations for LiSSA algorithm
'''


def get_inverse_hvp_lissa(model, criterion, dataset, vs,
                          batch_size,
                          scale=1,
                          damping=0.1,
                          num_repeats=1,
                          verbose=False):
    assert criterion is not None, "ERROR: Criterion cannot be None."
    assert batch_size <= len(dataset), "ERROR: Minibatch size for LiSSA should be less than dataset size"
    # assert len(dataset) % batch_size == 0, "ERROR: Dataset size for LiSSA should be a multiple of minibatch size"
    assert isinstance(dataset, Dataset), "ERROR: `dataset` must be PyTorch Dataset"

    params = [param for param in model.parameters() if param.requires_grad]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    inverse_hvp = None

    for rep in range(num_repeats):
        cur_estimate = vs
        for batch in iter(data_loader):
            batch_inputs, batch_targets = batch
            batch_out = model(batch_inputs)

            loss = criterion(batch_out, batch_targets) / batch_size

            hvp = hessian_vector_product(loss, params, vs=cur_estimate)
            cur_estimate = [v + (1 - damping) * ce - hv / scale \
                            for (v, ce, hv) in zip(vs, cur_estimate, hvp)]

        inverse_hvp = [hv1 + hv2 / scale for (hv1, hv2) in zip(inverse_hvp, cur_estimate)] \
            if inverse_hvp is not None \
            else [hv2 / scale for hv2 in cur_estimate]

    # avg. over repetitions
    inverse_hvp = [item / num_repeats for item in inverse_hvp] 
    return inverse_hvp



def inf_func(trnX, trnY, devX, devY, clf,
             epochs=5, trn_batch_size=1, dev_batch_size=16):

    if epochs > 0:
        print("Training models for IF with {} iterations ...".format(epochs))
        clf.fit(trnX, trnY, epochs=epochs, batch_size=trn_batch_size)

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

