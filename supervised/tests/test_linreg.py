import numpy as np 
import sys
sys.path.append('../linear')
from LinearRegression import LinearRegression

def test_sanity():
    X = np.array([[1,1],[2,2], [3,3]], dtype=np.float32).reshape(-1,2)
    y = np.array([3,5,7], dtype=np.float32).reshape(-1,1)
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred = lin_reg.predict(X)
    assert np.isclose(y, y_pred, rtol=0.1).any()

def test_ridge():
    X = np.array([[1,1],[2,2], [3,3]], dtype=np.float32).reshape(-1,2)
    y = np.array([3,5,7], dtype=np.float32).reshape(-1,1)
    lin_reg = LinearRegression()
    lin_reg.fit_ridge(X, y)
    y_pred = lin_reg.predict(X)
    assert np.isclose(y, y_pred, rtol=0.1).any()

def test_lasso():
    X = np.array([[1,1],[2,2], [3,3]], dtype=np.float32).reshape(-1,2)
    y = np.array([3,5,7], dtype=np.float32).reshape(-1,1)
    lin_reg = LinearRegression()
    lin_reg.fit_lasso(X, y)
    y_pred = lin_reg.predict(X)
    assert np.isclose(y, y_pred, rtol=0.1).any()