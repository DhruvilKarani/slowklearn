import numpy as np
import scipy
import math
import sys
sys.path.append('../')
from loss.functions import mse

class LinearRegression:
    def __init__():
        pass

    def fit(X, y, converge=0.001, max_runs=1000, lr=0.001):
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a numpy arra of shape (n_datapoints, n_features)")
        if not isinstance(y, np.ndarray):
            raise TypeError("y should be a numpy arra of shape (n_datapoints, 1)")

        coeffs = np.random.rand(1, X.shape[1])

        for run in range(max_runs):
            preds = np.matmul(coeffs, X)
            loss = mse(y, preds)
            grad = 



