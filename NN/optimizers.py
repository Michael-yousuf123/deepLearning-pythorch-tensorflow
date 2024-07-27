import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from crossmax import CrosMax
from neuralnet import DenseLayer, ReLU
class SGD:
    """
    """
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weight += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate*layer.dbiases
if __name__ == '__main__':
    X, y = spiral_data(samples=100, classes=3)
    dense1 = DenseLayer(2, 64)
    activation1 = ReLU()
    dense2 = DenseLayer(64, 3)
    loss_activation = CrosMax()
    optimizer = SGD()
    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        if not epoch % 100:
            print(f'epch: {epoch},' + f'acc: {accuracy:.3f}, '+ f'loss: {loss:.3f}')
        loss_activation.backward(loss_activation.output,y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)