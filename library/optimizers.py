class Optimizer:
    def update(self, key, weights, gradients):
        raise NotImplementedError("Update not implemented.")


class SGD(Optimizer):
    def __init__(self, learning_rate, momentum=0, dampening=0, weight_decay=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.velocity = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, " \
               f"momentum={self.momentum}, dampening={self.dampening}, " \
               f"weight_decay={self.weight_decay})"

    def update(self, key, weights, gradients):
        if self.weight_decay:
            gradients = gradients + (self.weight_decay * weights)
        if self.momentum:
            if key not in self.velocity.keys():
                self.velocity[key] = gradients
            else:
                self.velocity[key] = (self.momentum * self.velocity[key]) + ((1 - self.dampening) * gradients)
                gradients = self.velocity[key]
        return weights - self.learning_rate * gradients
