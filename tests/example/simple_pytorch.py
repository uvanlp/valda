## simple.py
## A simple example

import torch
from pickle import load
from sklearn import preprocessing

from valda.valuation import DataValuation
from valda.pyclassifier import PytorchClassifier


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        outputs = self.softmax(self.linear(x))
        return outputs

if __name__ == '__main__':
    data = load(open('../data/diabetes.pkl', 'rb'))
    trnX, trnY = data['trnX'], data['trnY']
    devX, devY = data['devX'], data['devY']
    print('trnX.shape = {}'.format(trnX.shape))

    labels = list(set(trnY))
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    trnY = le.transform(trnY)
    devY = le.transform(devY)


model = LogisticRegression(input_dim=trnX.shape[1], output_dim=len(labels))
clf = PytorchClassifier(model)

dv = DataValuation(trnX, trnY, devX, devY)

vals = dv.estimate(clf=clf, method='inf-func')

print(vals)
