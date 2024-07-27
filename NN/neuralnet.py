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
class ReLU:
    """
    """
    def forward(self, inputs):
        self.output =np.maximum(0, inputs)
class SoftMax:
    """
    """
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.maximum(inputs, axis = 1, keepdims=True))
        prob = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = prob
if __name__ == '__main__':
    # Datasets
    X, y = spiral_data(samples =100, classes=3)
    dense = DenseLayer(2, 3)
    activation = ReLU()
    dense.forward(X)
    activation.forward(dense.output)
    print(activation.output[:5])