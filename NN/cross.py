from loss import Loss
import numpy as np
class CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # number of samples in a batch
        samples = len(y_pred)
        # prevention of prediction divided by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # for one dimensional or sparse
        if len(y_true.shape) == 1:
            confidence = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape)==2:
            confidence = np.sum(y_pred_clipped*y_true, axis=1)
        loss_likelihood = -np.log(confidence)
        return loss_likelihood