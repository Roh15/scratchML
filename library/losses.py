import numpy as np


class MSE:
    def __init__(self):
        self.y = None
        self.y_hat = None

    def forward(self, y, y_hat):
        if y.shape != y_hat.shape:
            raise ValueError(f"Shape mismatch: got y shape: {y.shape} and y_hat shape: {y_hat.shape}")

        self.y = y
        self.y_hat = y_hat
        loss = np.mean(np.square(y - y_hat), axis=1)
        return loss.reshape(-1, 1)

    def backward(self):
        grad = -(2 / self.y.shape[1]) * np.sum(self.y - self.y_hat, axis=1)
        return grad.reshape(-1, 1)
