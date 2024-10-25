import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorlab.linear_model import LinearRegression


def test_simple_linear_data():
    np.random.seed(42)
    X = np.random.rand(100, 1)
    y = 5 * X.squeeze() + 3 + np.random.randn(100) * 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse < 0.2  # Check if the MSE is low enough

def test_zero_variation_data():
    X_zero_var = np.ones((10, 1))
    y_zero_var = np.ones(10) * 5
    model_zero = LinearRegression(lr=0.01, n_iters=1000)
    model_zero.fit(X_zero_var, y_zero_var)
    y_pred_zero = model_zero.predict(X_zero_var)
    assert np.allclose(y_pred_zero, y_zero_var)

def test_high_dimensional_data():
    X_high_dim = np.random.rand(100, 10)
    true_weights = np.array([2, -3, 0.5, 4, 0, -1, 2.5, -2, 1, -0.5])
    y_high_dim = X_high_dim.dot(true_weights) + 5 + np.random.randn(100) * 0.1
    X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(X_high_dim, y_high_dim, test_size=0.2, random_state=42)
    model_high_dim = LinearRegression(lr=0.01, n_iters=1000)
    model_high_dim.fit(X_train_hd, y_train_hd)
    y_pred_hd = model_high_dim.predict(X_test_hd)
    mse_hd = mean_squared_error(y_test_hd, y_pred_hd)
    assert mse_hd < 0.6  # MSE should be low
