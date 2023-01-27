## simple.py
## Date: 01/26/2023
## A simple example

from pickle import load
from eval import data_removal
from metrics import weighted_acc_drop


from valda.valuation import DataValuation


data = load(open('diabetes.pkl', 'rb'))

trnX, trnY = data['trnX'], data['trnY']
devX, devY = data['devX'], data['devY']


dv = DataValuation(trnX, trnY, devX, devY)


vals = dv.estimate(method='beta-shapley')


tstX, tstY = data['tstX'], data['tstY']

accs = data_removal(vals, trnX, trnY, tstX, tstY)
res = weighted_acc_drop(accs)
print("The weighted accuracy drop is {}".format(res))
