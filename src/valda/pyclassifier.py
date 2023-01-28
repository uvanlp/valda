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
            self.y = [-1]*self.X.shape[0]
        self.len = self.X.shape[0]
        self.dim = self.X.shape[1]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

    def __dim__(self):
        return self.dim


class PytorchClassifier(object):
    def __init__(self, model, optim=None, loss=None,
                     epochs=10,
                     trn_batch_size=1,
                     dev_batch_size=16):
        '''
        model - a classifier built with PyTorch
        optim - a optimizer defined in PyTorch
        loss - the loss function for training
        epochs - the number of epochs for training (epochs=0 for no training)
        trn_batch_size - mini-batch size for training
        dev_batch_size - mini-batch size for evaluation
        '''
        self.model = model
        self.model_state_dict = model.state_dict()
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
        self.epochs = epochs
        self.trn_batch_size = trn_batch_size
        self.dev_batch_size = dev_batch_size


    def fit(self, X, y):
        '''
        X, y - training examples
        '''
        # Re-load the model before training
        self.model.load_state_dict(self.model_state_dict)
        loader = DataLoader(Data(X,y), batch_size=self.trn_batch_size,
                            shuffle=True, num_workers=0)
        for epoch in range(self.epochs):
            for (inputs, labels) in loader:
                self.optim.zero_grad()
                outputs = self.model(inputs)
                batch_loss = self.loss(outputs, labels)
                batch_loss.backward()
                self.optim.step()
        # print("Done training")


    def predict(self, X):
        data = Data(X)
        loader = DataLoader(data, batch_size=self.dev_batch_size,
                            shuffle=False,
                            num_workers=0)
        pred_labels = []
        with torch.no_grad():
            for data in loader:
                inputs, _ = data
                outputs = self.model(inputs)
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
        idx = 0
        for (inputs, labels) in loader:
            idx += 1
            outputs = self.model(inputs)
            batch_loss = self.loss(outputs, labels)
            batch_grads = grad(batch_loss, self.params)
            grads.append([batch_grads]) # are the data structures consistent?
        return grads
        

        
    def get_parameters(self):
        ''' Get the model parameters
        '''
        return self.params
