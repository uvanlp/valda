## classifier.py
## Date: 01/18/2023
## A general framework of defining a classifier class


import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class Data(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X.astype(np.float32))
        if y is not None:
            self.y = torch.from_numpy(y).type(torch.LongTensor)
        else:
            self.y = None
        self.len = self.X.shape[0]
        self.dim = self.X.shape[1]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

    def __dim__(self):
        return self.dim


class PytorchClassifier(object):
    def __init__(self, model, optim=None, loss=None):
        '''
        model - 
        optim - 
        loss - 
        '''
        self.model = model
        self.params = list(self.model.parameters())
        # optimizer
        if optim is None:
            self.optim = torch.optim.Adam(model.parameters())
        else:
            self.optim = optim
        # loss function
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss


    def fit(self, X, y, epochs=10, batch_size=1):
        '''
        X, y - training examples
        '''
        loader = DataLoader(Data(X,y), batch_size=batch_size, shuffle=True,
                            num_workers=0)
        for epoch in tqdm(range(epochs)):
            for (inputs, labels) in loader:
                self.optim.zero_grad()
                outputs = self.model(inputs)
                batch_loss = self.loss(outputs, labels)
                batch_loss.backward()
                self.optim.step()
        print("Done training")


    def predict(self, X):
        data = Data(X)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            num_workers=0)
        pred_labels = []
        with torch.no_grad():
            for data in loader:
                inputs, _ = data
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                pred_labels += predicted
        return pred_labels


    def online_train(self, x, y):
        ''' On line training
        '''
        raise NotImplementedError("Not implemented")


    def grad(self, X, y, batch_size=1):
        ''' Compute the gradient of the parameter wrt (X, y)
        '''
        grads = []
        loader = DataLoader(Data(X,y), batch_size=batch_size, shuffle=False,
                            num_workers=0)
        for (inputs, labels) in loader:
            outputs = self.model(inputs)
            batch_loss = self.loss(outputs, labels)
            batch_grads = grad(batch_loss, self.params)
            grads += batch_grads # are the data structures consistent?
        return grads
        

        
    def get_parameters(self):
        ''' Get the model parameters
        '''
        return self.params
