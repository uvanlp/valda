## params.py
## Date: 01/23/2022
## Initialize default parameters


class Parameters(object):
    def __init__(self):
        self.params = {
            # For TMC Shapley
            'tmc_iter':500,
            'tmc_thresh':0.001,
            # For CS Shapley
            'cs_iter':500,
            'cs_thresh':0.001,
            # For Beta Shapley
            'beta_iter':50,
            'alpha':1.0,
            'beta':16.0,
            'rho':1.0005,
            'beta_chain':10,
            # For Influence Function
            'if_iter':30,
            'trn_batch_size':16,
            'dev_batch_size':16
            }
    
    def update(self, new_params):
        for (key, val) in new_params.items():
            try:
                self.params[key] = val
            except KeyError:
                print("Undefined key {} with value {}".format(key))
        # return self.params


    def get_values(self):
        return self.params


    def print_values(self):
        print("The current hyper-parameter setting:")
        for (key, val) in self.params.items():
            print("\t{} : {}".format(key, val))
