import numpy as np 

def mse(y_true, y_pred):
    '''
    --Computes mean squared error between two numpy arrays

    --params:
        y_true: ground truth labels
        y_pred: predicted labels

    --returns:
        mse
    '''
    y_true = np.array(y_true).reshape(len(y_true), 1)
    y_pred = np.array(y_pred).reshape(len(y_pred), 1)

    assert y_true.shape == y_pred.shape, "Shape of predicted and true labels is unequal"

    return np.mean(np.power(y_true-y_pred, 2))
