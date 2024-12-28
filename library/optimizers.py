class Optimizer:
    def update(self, weights, gradients):
        raise NotImplementedError("Update not implemented.")


class SGD(Optimizer):
    def __init__(self, learning_rate, momentum=0, dampening=0, weight_decay=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.velocity = None

    def __repr__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, " \
               f"momentum={self.momentum}, dampening={self.dampening}, " \
               f"weight_decay={self.weight_decay})"

    def update(self, weights, gradients):
        if self.weight_decay:
            gradients = gradients + (self.weight_decay * weights)
        if self.momentum:
            if self.velocity is None:
                self.velocity = gradients
            else:
                self.velocity = (self.momentum * self.velocity) + ((1 - self.dampening) * gradients)
                gradients = self.velocity
        return weights - self.learning_rate * gradients
