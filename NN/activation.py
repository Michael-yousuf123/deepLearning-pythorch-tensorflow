import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
class SoftMax:
    def forward(self, inputs):
        # get exponential values of each
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        prob = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = prob
X, y = spiral_data(samples=100, classes=3)
dense = DenseLayer(2, 3)
dense2 = DenseLayer(3, 3)
relu = ReLU()
softmax = SoftMax()

# Make a forward pass of our training data through this layer
dense.forward(X)
# Make a forward pass through activation function
# it takes the output of first dense layer here
relu.forward(dense.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(relu.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
softmax.forward(dense2.output)
# Let's see output of the first few samples:
print(softmax.output[:5])