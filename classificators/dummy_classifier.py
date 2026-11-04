import numpy as np

class DummyClassifier:
    def __init__(self, classes):
        self.classes = classes

    def predict(self, test_X):
        # sample a vector of zeros and ones
        predictions = np.random.randint(low=0, high=2, size=(len(test_X), len(self.classes)))
        return predictions

    def train(self, train, val):
        pass