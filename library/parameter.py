import numpy as np


class Parameter:
    def __init__(self, params, optimizer):
        self._params = np.array(params)  # Ensure params are stored as a NumPy array
        self._optimizer = optimizer

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self._params.shape}, optimizer={self._optimizer})"

    def __array__(self, dtype=None):
        return np.array(self._params, dtype=dtype)

    def step(self, grad):
        self._params = self._optimizer.update(self._params, grad)

    @property
    def shape(self):
        return self._params.shape

    @property
    def T(self):
        return Parameter(self._params.T, self._optimizer)

    # Arithmetic operations (only essential ones)
    def __add__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self._params + other._params, self._optimizer)
        else:
            return Parameter(self._params + other, self._optimizer)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self._params * other._params, self._optimizer)
        else:
            return Parameter(self._params * other, self._optimizer)

    def __rmul__(self, other):
        return self.__mul__(other)

    # Inplace operators.
    def __iadd__(self, other):
        if isinstance(other, Parameter):
            self._params += other._params
        else:
            self._params += other
        return self

    def __imul__(self, other):
        if isinstance(other, Parameter):
            self._params *= other._params
        else:
            self._params *= other
        return self

    # Getters
    def to_numpy(self):
        return self._params
