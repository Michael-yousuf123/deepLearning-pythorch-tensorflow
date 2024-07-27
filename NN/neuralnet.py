import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class DenseLayer:
    def __init__(self, ninputs, nneurons):
        """
        """
        self.weight = np.random.randn(ninputs, nneurons)
        self.biases = np.zeros((1, nneurons))
    def forward(self, inputs):
        """
        """
        self.output = np.dot(inputs, self.weight) + self.biases
if __name__ == '__main__':
    # Datasets
    X, y = spiral_data(samples =100, classes=3)
    dense = DenseLayer(2, 3)
    dense.forward(X)
    print(dense.output[:5])