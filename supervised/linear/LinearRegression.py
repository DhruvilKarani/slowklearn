import numpy as np
import scipy
import math
import sys
sys.path.append('../../')
from loss.functions import mse

class LinearRegression:
    def __init__(self):
        return

    def fit(self, X, y, converge=0.00001, max_runs=100, lr=0.001):
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a numpy arra of shape (n_datapoints, n_features)")
        if not isinstance(y, np.ndarray):
            raise TypeError("y should be a numpy arra of shape (n_datapoints, 1)")

        coeffs = np.random.rand(1, X.shape[1])
        N = X.shape[0]
        for run in range(max_runs):
            preds = np.matmul(X, coeffs.T)
            loss = mse(y, preds)
            grad = np.zeros_like(X[0:1])
            # print(grad.shape)
            for i in range(N):
                x = X[i].reshape(1,-1)
                res = (y[i] - np.matmul(x, coeffs.T)).reshape(-1,1)
                grad += np.matmul(res, x)
            print(run,"Loss: ",loss,"Grad: ", grad)
            grad *= -2/N            
            coeffs -= lr*grad
        print(preds)
        return coeffs

if __name__ == '__main__':
    X = np.array([[1,1],[2,2]], dtype=np.float32).reshape(2,2)
    y = np.array([2,4], dtype=np.float32).reshape(2,1)
    lin_reg = LinearRegression()
    print(lin_reg.fit(X, y))


