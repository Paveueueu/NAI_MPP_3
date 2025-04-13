from src.perceptron import Perceptron

def normalize_vector(vector):
    import math
    dist = math.sqrt(sum(x**2 for x in vector))
    if dist > 0:
        return [x / dist for x in vector]
    return vector

class SingleLayerClassifier:
    def __init__(self, languages, learning_rate, bias):
        self.neurons = {language: Perceptron(26, learning_rate, bias) for language in languages}

    def predict_class(self, data):
        X = normalize_vector(data["X"])
        nets = {}
        for neuron_class, neuron in self.neurons.items():
            net = neuron.net(X)
            nets[neuron_class] = net

        return max(nets, key=nets.get)


    def learn(self, learnData):
        for data in learnData:
            X = normalize_vector(data["X"])
            for neuron_class, neuron in self.neurons.items():
                desired = (neuron_class == data["class"])
                neuron.learn(X, desired)

    def test(self, testData):
        result = []
        for data in testData:
            result.append({'class': data['class'], 'prediction': self.predict_class(data), 'name': data['name']})
        return result



