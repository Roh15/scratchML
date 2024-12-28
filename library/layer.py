import numpy as np
from library.parameter import Parameter


class Layer:
    def forward(self, x):
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, grad):
        raise NotImplementedError("Backward pass not implemented.")


class Linear(Layer):
    def __init__(self, input_dim, output_dim, bias, optimizer, initialization='He'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        if initialization == 'Xavier':
            # Recommended for tanh or sigmoid activation
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.weights = Parameter(params=np.random.uniform(-limit, limit, (input_dim, output_dim)),
                                     optimizer=optimizer)  # Weights matrix
        elif initialization == 'He':
            # Recommended for ReLU activation
            std = np.sqrt(2 / input_dim)
            self.weights = Parameter(params=np.random.normal(0, std, (input_dim, output_dim)),
                                     optimizer=optimizer)  # Weights matrix
        elif initialization == 'LeCun':
            # Recommended for SELU activation
            std = np.sqrt(1 / input_dim)
            self.weights = Parameter(params=np.random.normal(0, std, (input_dim, output_dim)),
                                     optimizer=optimizer)  # Weights matrix
        else:
            # Default is truncated random
            weights = np.random.normal(0, 1, (input_dim, output_dim))
            weights = np.clip(weights, -2, 2)
            self.weights = Parameter(params=weights,
                                     optimizer=optimizer)  # Weights matrix
        if bias:
            self.biases = Parameter(params=np.zeros(output_dim),
                                    optimizer=optimizer)  # Bias vector
        else:
            self.biases = None
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
        self.weights.step(grad_w)
        if self.bias:
            grad_b = np.mean(grad, axis=0)
            self.biases.step(grad_b)

        grad = np.dot(grad, self.weights.T)
        return grad
