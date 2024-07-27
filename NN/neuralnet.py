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
        self.inputs = inputs
        self.output = np.dot(inputs, self.weight) + self.biases
    def backward(self, dvalues):
        """
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weight.T)
class ReLU:
    """
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output =np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftMax:
    """
    """
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.maximum(inputs, axis = 1, keepdims=True))
        prob = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = prob
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_output= single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
if __name__ == '__main__':
    # Datasets
    X, y = spiral_data(samples =100, classes=3)
    dense = DenseLayer(2, 3)
    dense2 = DenseLayer(3, 3)
    activation = ReLU()
    activation2 = SoftMax()
    dense.forward(X)
    activation.forward(dense.output)
    dense2.forward(activation.output)
    activation2.forward(dense2.output)
    print(activation2.output[:5])