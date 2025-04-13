class Perceptron:
    def __init__(self, size, learning_rate, bias):
        self.learning_rate = learning_rate

        import random
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(size)]

        self.bias = bias
        self.size = size

    def normalize_weights(self):
        import math
        dist = math.sqrt(sum(w ** 2 for w in self.weights))
        if dist > 0:
            self.weights = [w / dist for w in self.weights]

    def learn(self, X, desired):
        if (prediction := self.predict(X)) == desired:
            return True
        error = (1 if desired else 0) - (1 if prediction else 0)
        for i in range(self.size):
            self.weights[i] += self.learning_rate * error * X[i]
        self.bias += self.learning_rate * error
        self.normalize_weights()
        return False

    def net(self, X):
        net = self.bias
        for i in range(self.size):
            net += self.weights[i] * X[i]
        return net

    def predict(self, X):
        return self.net(X) > 0
