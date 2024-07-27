import numpy as np
import nnfs
from nnfs import spiral_data

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
