import numpy as np
class Loss:
    def calculate(self, output, y):
        #sample loss
        sample_loss = self.forward(output, y)
        #data loss
        data_loss = np.mean(sample_loss)
        return data_loss