import numpy as np
import scipy
import math
import sys
sys.path.append('../../')
from loss.functions import mse

class LinearRegression:
    def __init__(self):
        self.coeffs_ = None

    def fit(self, X, y, converge=10e-10, max_runs=1000, lr=0.01):
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a numpy arra of shape (n_datapoints, n_features)")
        if not isinstance(y, np.ndarray):
            raise TypeError("y should be a numpy arra of shape (n_datapoints, 1)")

        M = X.shape[1]
        N = X.shape[0]
        X = np.column_stack((X, np.ones((N, 1), dtype=np.float32)))
        coeffs = np.random.rand(1, M+1)
        for run in range(max_runs):
            preds = np.matmul(X, coeffs.T)
            loss = mse(y, preds)
            res = (y - preds).T
            grad = np.matmul(res, X)
            # print(run,"Loss: ",loss,"Grad: ", grad)
            grad *= -2/N
            coeffs -= lr*grad
            if np.mean(np.abs(grad)) < converge:
                break
        self.coeffs_ = coeffs
        return coeffs


    def fit_ridge(self, X, y, converge=10e-10, max_runs=1000, lr=0.01, C=1.0):
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a numpy arra of shape (n_datapoints, n_features)")
        if not isinstance(y, np.ndarray):
            raise TypeError("y should be a numpy arra of shape (n_datapoints, 1)")

        M = X.shape[1]
        N = X.shape[0]
        X = np.column_stack((X, np.ones((N, 1), dtype=np.float32)))
        coeffs = np.random.rand(1, M+1)
        for run in range(max_runs):
            preds = np.matmul(X, coeffs.T)
            loss = mse(y, preds)
            res = (y - preds).T
            grad = np.matmul(res, X)
            # print(run,"Loss: ",loss,"Grad: ", grad)
            grad *= -2/N
            grad += C*2*coeffs
            coeffs -= lr*grad
            if abs(np.mean(grad)) < converge:
                break
        self.coeffs_ = coeffs
        return coeffs


    def fit_lasso(self, X, y, converge=10e-10, max_runs=1000, lr=0.01, C=1.0):
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a numpy arra of shape (n_datapoints, n_features)")
        if not isinstance(y, np.ndarray):
            raise TypeError("y should be a numpy arra of shape (n_datapoints, 1)")

        M = X.shape[1]
        N = X.shape[0]
        X = np.column_stack((X, np.ones((N, 1), dtype=np.float32)))
        coeffs = np.random.rand(1, M+1)
        for run in range(max_runs):
            preds = np.matmul(X, coeffs.T)
            loss = mse(y, preds)
            res = (y - preds).T
            grad = np.matmul(res, X)
            # print(run,"Loss: ",loss,"Grad: ", grad)
            grad *= -2/N
            lasso_comp = coeffs.copy()
            lasso_comp[lasso_comp>0]=1
            lasso_comp[lasso_comp<0] = -1
            grad += C*lasso_comp
            coeffs -= lr*grad
            if abs(np.mean(grad)) < converge:
                break
        self.coeffs_ = coeffs
        return coeffs


    def predict(self, X):
        bias = self.coeffs_.T[-1]
        weights = self.coeffs_.T[:-1]
        return  np.matmul(X, weights) + bias

    
    

if __name__ == '__main__':
    X = np.array([[1,1],[2,2], [3,3]], dtype=np.float32).reshape(-1,2)
    y = np.array([3,5,7], dtype=np.float32).reshape(-1,1)
    lin_reg = LinearRegression()
    print(lin_reg.fit(X, y))
    print(lin_reg.predict(X))
    print(lin_reg.fit_lasso(X, y))
    print(lin_reg.predict(X))
    print(lin_reg.fit_lasso(X, y, C=10))
    print(lin_reg.predict(X))


