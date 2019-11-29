import numpy as np
import scipy
import math
import sys
sys.path.append('../../')
import sklearn
from sklearn import datasets

class BinaryLogisticRegression:

    def __init__(self):
        self.coeffs_ = None
    
    @staticmethod
    def prob(affine_trans):
        return 1/(1+np.exp(-affine_trans))

    @staticmethod
    def affine(x, coeffs):
        return np.matmul(coeffs, x)
    
    def fit(self, X, y, max_runs=100, converge=0.001, lr=10e-3, C=1.0):
        N = X.shape[0]
        M = X.shape[1]
        X = np.column_stack((X, np.ones((N, 1), dtype=np.float32)))
        y = y.reshape(-1,1)
        coeffs = np.random.rand(1, M+1)
        for run in range(max_runs):
            probs = np.apply_along_axis(lambda x: self.prob(self.affine(x, coeffs)), 1, X)
            diff = (y-probs).reshape(1,-1)
            grad = np.matmul(diff, X)/N
            coeffs += lr*grad
            if np.mean(np.abs(grad))<converge:
                break
        
        self.coeffs_ = coeffs

    def proba_(self, X):
        N = X.shape[0]
        M = X.shape[1]
        X = np.column_stack((X, np.ones((N, 1), dtype=np.float32)))
        probs = np.apply_along_axis(lambda x: self.prob(self.affine(x, self.coeffs_)), 1, X)
        return probs


    def predict(self, X, thresh=0.5):
        probs = self.proba_(X)
        probs[probs>=thresh] = 1
        probs[probs<thresh] = 0
        return probs.reshape(-1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y)/y.shape[0]
    


if __name__ == '__main__':
    log_reg = BinaryLogisticRegression()
    iris = datasets.load_iris()
    X = iris.data[:80]
    y = iris.target[:80]
    log_reg.fit(X,y)
    y_pred = log_reg.predict(X)
    print(log_reg.score(X, y))