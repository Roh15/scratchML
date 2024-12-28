import numpy as np


class Activation:
    def forward(self, x):
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, grad):
        raise NotImplementedError("Backward pass not implemented.")


class ReLU(Activation):
    """
    ReLU(x) = x if x > 0
              0 if x <= 0
    """
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * np.where(self.input > 0, 1, 0)


class Tanh(Activation):
    """
    Tanh(x) = tanh(x)
    Bound between [-1, 1]
    """
    def __init__(self):
        self.tanh = None

    def forward(self, x):
        self.tanh = np.tanh(x)
        return self.tanh

    def backward(self, grad):
        return grad * (1 - np.square(self.tanh))


class Sigmoid(Activation):
    """
    Sigmoid(x) = 1/1 + e^-x
    Bound [0, 1]
    """
    def __init__(self):
        self.sigmoid = None

    def forward(self, x):
        self.sigmoid = 1/(1 + np.exp(-x))
        return self.sigmoid

    def backward(self, grad):
        return grad * self.sigmoid * (1 - self.sigmoid)


class Swish(Activation):
    """
    Swish(x) = x * (1/1 + e^-x)
    """
    def __init__(self):
        self.input = None
        self.sigmoid = None

    def forward(self, x):
        self.input = x
        self.sigmoid = Sigmoid().forward(x)
        return x * self.sigmoid

    def backward(self, grad):
        sigmoid_derivative = self.sigmoid * (1 - self.sigmoid)
        return grad * (self.sigmoid + (self.input * sigmoid_derivative))


class SELU(Activation):
    """
    SELU(x) = scale * (max(0,x)+min(0,alpha*(exp(x)−1)))
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    """
    def __init__(self):
        self.input = None
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def forward(self, x):
        self.input = x
        return self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, grad):
        return grad * self.scale * np.where(self.input > 0, 1, self.alpha * np.exp(self.input))
