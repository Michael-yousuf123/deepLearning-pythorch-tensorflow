from softmax import SoftMax
from cross import CrossEntropy

class CrosMax():
    # creates activation and loss function instant objects
    def __init__(self):
        self.activation = SoftMax()
        self.loss = CrossEntropy()
    # forward pass
    def forward(self, inputs, y_true):
        #output layer activation function
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        #calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        #if values are one hot-encoded
        #turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # copy so we can modify 
        self.dinputs = dvalues.copy()
        # calculate gradients
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs =self.dinputs/samples