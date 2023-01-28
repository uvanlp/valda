## simple.py
## Date: 01/26/2023
## A simple example

import valda

from pickle import load
from valda.eval import data_removal
from valda.metrics import weighted_acc_drop
from valda.valuation import DataValuation


if __name__ == '__main__':
    data = load(open('diabetes.pkl', 'rb'))

    trnX, trnY = data['trnX'], data['trnY']
    devX, devY = data['devX'], data['devY']

    dv = DataValuation(trnX, trnY, devX, devY)

    vals = dv.estimate(method='loo')

    tstX, tstY = data['tstX'], data['tstY']

    accs = data_removal(vals, trnX, trnY, tstX, tstY)
    res = weighted_acc_drop(accs)
    print("The weighted accuracy drop is {}".format(res))
