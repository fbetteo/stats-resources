# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

# Multiclass Classification using PyTorch for tabular data.


from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch import long
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


# dataset definition
# We create a class to handle the data

""" It's standard in PyTorch to extend Dataset class
# PyTorch provides the Dataset class that you can extend and customize to load your dataset.
# For example, the constructor of your dataset object can load your data file (e.g. a CSV file). You can then override the __len__() function that can be used to get the length of the dataset (number of rows or samples), and the __getitem__() function that is used to get a specific sample by index.
# When loading your dataset, you can also perform any required transforms, such as scaling or encoding.
# Once loaded, PyTorch provides the DataLoader class to navigate a Dataset instance during the training and evaluation of your model.
# A DataLoader instance can be created for the training dataset, test dataset, and even a validation dataset."""


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is float
        self.X = self.X.astype('float32')
        # label enconde target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)

    # number of rows in the dataset

    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# Model Definition
""" Inherits from Module
The trainig is callable (probably something complex that usets forward())
The output (yhat) is derivable and keeps information I guess since loss object calculate the derivatives

 """


class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 3)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# prepare the dataset


def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate splits
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()  # binary log loss / crossentropy
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(500):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            targets = targets.type('torch.LongTensor')
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model in the test set
        yhat = model(inputs)  # model es de clase MLP
        # retrieve numpy array
        yhat = yhat.detach().numpy()  # tensor to array
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data


def predict(row, model):
    # convert row to data
    row = Tensor([row])  # ?
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(4)
# train the model
train_model(
    train_dl, model)  # not sure how the optimizer works but the funcion doesn't return\
# anything, somehow modifying inplace
# evaluate the model
acc = evaluate_model(test_dl, model)
print(f'Accuracy: {acc:.3%}')
# make a single prediction (expect class=1)
row = [5.1, 3.5, 1.4, 0.2]
yhat = predict(row, model)
print(f'Predicted {yhat[0][0]:.3%} (class={argmax(yhat)}')
