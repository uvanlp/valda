## valuation.py
## Date: 01/18/2023
## A general framework of defining data valuation class


from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score


## Local module: load data valuation methods
from .loo import loo
from .tmc_shapley import tmc_shapley
from .cs_shapley import cs_shapley
from .beta_shapley import beta_shapley
from .inf_func import inf_func
from .params import Parameters # Import default model parameters


class DataValuation(object):
    def __init__(self, trnX, trnY, devX=None, devY=None):
        '''
        trn_X, trn_Y - Input/output for training, also the examples for 
                       being valued
        val_X, val_Y - Input/output for validation, also the examples used 
                       for estimating the values of (trn_X, trn_Y)
        '''
        self.trnX, self.trnY = trnX, trnY
        if devX is None:
            self.valX, self.devY = trnX, trnY
        else:
            self.devX, self.devY = devX, devY
        self.values = {} # A rank list of
        self.clf = None # instance of classifier
        params = Parameters()
        self.params = params.get_values()


    def estimate(self, clf=None, method='loo', params=None):
        '''
        clf - a classifier instance (Logistic regression, by default)
        method - the data valuation method (LOO, by default)
        params - hyper-parameters for data valuation methods
        '''
        self.values = {}
        if clf is None:
            self.clf = LR(solver="liblinear", max_iter=500, random_state=0)
        else:
            self.clf = clf

        if params is not None:
            print("Overload the model parameters with the user specified ones: {}".format(params))
            self.params = params

        # Call data valuation functions
        if method == 'loo':
            # Leave-one-out
            vals = loo(self.trnX, self.trnY, self.devX, self.devY, self.clf)
            for idx in range(len(vals)):
                self.values[idx] = vals[idx]
        elif method == 'tmc-shapley':
            # TMC Data Shapley (TODO: Citation)
            # Get the default parameter values
            n_iter = self.params['tmc_iter']
            tmc_thresh = self.params['tmc_thresh']
            # 
            vals = tmc_shapley(self.trnX, self.trnY, self.devX, self.devY,
                                   self.clf, n_iter, tmc_thresh)
            for idx in range(len(vals)):
                self.values[idx] = vals[idx]
        elif method == 'cs-shapley':
            # CS Shapley (Schoch et al., 2022)
            n_iter = self.params['cs_iter']
            cs_thresh = self.params['cs_thresh']
            labels = list(set(self.trnY))
            for label in labels:
                vals, orig_indices = cs_shapley(self.trnX, self.trnY, self.devX,
                                                self.devY, label,
                                                self.clf, n_iter, cs_thresh)
                for (idx, val) in zip(list(orig_indices), list(vals)):
                    self.values[idx] = val
        elif method == 'beta-shapley':
            # Beta Shapley
            n_iter = self.params['beta_iter']
            alpha, beta = self.params['alpha'], self.params['beta']
            rho = self.params['rho']
            n_chain = self.params['beta_chain']
            vals = beta_shapley(self.trnX, self.trnY, self.devX, self.devY,
                                    self.clf, alpha, beta, rho, n_chain, n_iter)
            for idx in range(len(vals)):
                self.values[idx] = vals[idx]
        elif method == 'inf-func':
            n_iter = self.params['if_iter']
            trn_batch_size = self.params['trn_batch_size']
            dev_batch_size = self.params['dev_batch_size']
            vals = inf_func(self.trnX, self.trnY, self.devX, self.devY,
                                clf=self.clf)
            for idx in range(len(vals)):
                self.values[idx] = vals[idx]
        else:
            raise ValueError("Unrecognized data valuation method: {}".format(method))
        return self.values
    
    
    def get_values(self):
        '''
        return the data values
        '''
        if self.values is not None:
            return self.values
        else:
            raise ValueError("No values computed")
