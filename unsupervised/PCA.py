'''
    --Implementation of PCA
'''
import numpy as np
from numpy.linalg import eig


class PCA:

    def __init__(self, n_components):
        self.n = n_components
        self.axes_ = None
        self.explained_variance_ = None

    @staticmethod
    def _covariance_matrix(X):
        return np.cov(X.T)

    def fit(self, X):
        M = X.shape[0]
        N = X.shape[1]
        eig_val, eig_vect = eig(self._covariance_matrix(X))
        idxes = (-eig_val).argsort()[:self.n]
        self.axes_ = np.array([eig_vect[:, idx] for idx in idxes])
        self.explained_variance_ = sorted(eig_val, reverse=True)
        
    def transform(self, X):
        return np.matmul(self.axes_.T, X)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

