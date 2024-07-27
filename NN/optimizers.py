import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from crossmax import CrosMax
class SGD:
    """
    """
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate*layer.dbiases
if __name__ == '__main__':
    optimizer = SGD()
    optimizer.update_params()
