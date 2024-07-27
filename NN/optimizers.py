import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from crossmax import CrosMax
from neuralnet import DenseLayer, ReLU
class SGD:
    """
    """
    def __init__(self, learning_rate=1, decay = 0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.momentum = momentum
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1/(1+self.decay *self.iteration))
    def update_params(self, layer):
        """
        """
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weight)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentum = bias_updates
        else:
            layer.weight += -self.current_learning_rate * layer.dweights
            layer.biases += -self.current_learning_rate*layer.dbiases
        layer.weight += weight_updates
        layer.biases += bias_updates
    def post_update_params(self):

        self.iteration += 1
if __name__ == '__main__':
    X, y = spiral_data(samples=100, classes=3)
    dense1 = DenseLayer(2, 64)
    activation1 = ReLU()
    dense2 = DenseLayer(64, 3)
    loss_activation = CrosMax()
    optimizer = SGD(decay = 1e-3, momentum=0.5)
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
            print(f'epoch: {epoch},' + f'acc: {accuracy:.3f}, '+ f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate:.3f}')
        loss_activation.backward(loss_activation.output,y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()