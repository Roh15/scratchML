from library.layer import *
from library.losses import *
from library.activations import *
from library.optimizers import *
from library.sequential import Sequential
from library.data_loader import DataLoader

from copy import deepcopy

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class FeedForwardNN(Sequential):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 loss_fn,
                 optimizer):
        super().__init__(loss_fn)
        self.layers = [
            Linear(input_dim, hidden_dim, bias=True, optimizer=deepcopy(optimizer)),
            Swish(),
            Linear(hidden_dim, output_dim, bias=True, optimizer=deepcopy(optimizer)),
            Sigmoid()
        ]
        self.optimizer = optimizer


# load the data
X, y = load_diabetes(return_X_y=True)
y = MinMaxScaler().fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

train_loader = DataLoader(X_train, y_train, 64)
val_loader = DataLoader(X_val, y_val, 32, shuffle=False)
test_loader = DataLoader(X_test, y_test, 32, shuffle=False)

# initialize
ffnn = FeedForwardNN(input_dim=10, hidden_dim=5, output_dim=1,
                     loss_fn=MSE(),
                     optimizer=SGD(learning_rate=1e-3))

# train
ffnn.train(epochs=10, train_loader=train_loader, val_loader=val_loader)

# test
ffnn.test(test_loader=test_loader)
