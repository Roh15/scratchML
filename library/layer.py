import numpy as np


class Layer:
    def forward(self, x):
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, grad):
        raise NotImplementedError("Backward pass not implemented.")


class Linear(Layer):
    def __init__(self, input_dim, output_dim, optimizer, bias=True, initialization='He'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.bias = bias
        if initialization == 'Xavier':
            # Recommended for tanh or sigmoid activation
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif initialization == 'He':
            # Recommended for ReLU activation
            std = np.sqrt(2 / input_dim)
            self.weights = np.random.normal(0, std, (input_dim, output_dim))
        elif initialization == 'LeCun':
            # Recommended for SELU activation
            std = np.sqrt(1 / input_dim)
            self.weights = np.random.normal(0, std, (input_dim, output_dim))
        else:
            # Default is truncated random
            weights = np.random.normal(0, 1, (input_dim, output_dim))
            self.weights = np.clip(weights, -2, 2)
        if bias:
            self.biases = np.zeros(output_dim)
        else:
            self.biases = None
        self._weights_key = f'{id(self)}-weights'
        self._biases_key = f'{id(self)}-biases'
        self.input = None

    def forward(self, x):
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input shape mismatch: expected {self.input_dim}, got {x.shape[1]}")
        self.input = x
        out = np.dot(x, self.weights)
        if self.bias:
            out = out + self.biases
        return out

    def backward(self, grad):
        grad_w = np.dot(self.input.T, grad)
        self.weights = self.optimizer.update(self._weights_key, self.weights, grad_w)
        if self.bias:
            grad_b = np.mean(grad, axis=0)
            self.biases = self.optimizer.update(self._biases_key, self.biases, grad_b)

        grad = np.dot(grad, self.weights.T)
        return grad
