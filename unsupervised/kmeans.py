import numpy as np 
import scipy
from scipy.spatial import distance

class KMeans:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.means_ = None

    @staticmethod
    def distance(a, b):
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        print("a, b:", a,b)
        return distance.euclidean(a, b)

    def closest(self, point, means):
        distances = [self.distance(point, mean) for mean in means]
        print("D:", distances)
        assert distances != []
        return np.argmax(distances), distances

    def fit(self, X, n_iter=100, converge=10e-4):
        N = X.shape[0]
        M = X.shape[1]
        means = np.random.randn(self.n_clusters, M)
        y = np.zeros(N)
        for n in range(n_iter):
            total_mean_distance = 0
            print("means: ", means)
            for i, point in enumerate(X):
                y[i], distances = self.closest(point, means)
                print("closest: ", y[i])
                total_mean_distance += np.mean(distances)

            for j, mean in enumerate(means):
                if len(X[y.ravel()==j])>0:
                    means[j] = np.mean(X[y.ravel()==j], axis=0)
                    print("y: ",y, X[y.ravel()==j])
            
            self.means_ = means
            if total_mean_distance/N < converge:
                break
        
    def predict(self, X):
        y = np.zeros(N)
        for i, point in enumerate(X):
            y[i], _ = self.closest(point, means)
        
        return y
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds.ravel() == y.ravel())

if __name__ == '__main__':
    km = KMeans(1)
    X = np.array([[1,0,0], [0,1,0], [0,0,1]])
    km.fit(X)
    print(km.means_)