## simple.py
## A simple example

import torch
from pickle import load
from sklearn import preprocessing

import valda
from valda.valuation import DataValuation
from valda.pyclassifier import PytorchClassifier
from valda.eval import data_removal
from valda.metrics import weighted_acc_drop


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        outputs = self.softmax(self.linear(x))
        return outputs

if __name__ == '__main__':
    data = load(open('diabetes.pkl', 'rb'))
    trnX, trnY = data['trnX'], data['trnY']
    devX, devY = data['devX'], data['devY']
    tstX, tstY = data['tstX'], data['tstY']
    print('trnX.shape = {}'.format(trnX.shape))

    labels = list(set(trnY))
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    trnY = le.transform(trnY)
    devY = le.transform(devY)
    tstY = le.transform(tstY)

    model = LogisticRegression(input_dim=trnX.shape[1], output_dim=len(labels))
    clf = PytorchClassifier(model, epochs=20, trn_batch_size=16,
                                dev_batch_size=16)

    dv = DataValuation(trnX, trnY, devX, devY)

    vals = dv.estimate(clf=clf, method='loo')

    # print(vals)

    accs = data_removal(vals, trnX, trnY, tstX, tstY)
    res = weighted_acc_drop(accs)
    print("The weighted accuracy drop is {}".format(res))
