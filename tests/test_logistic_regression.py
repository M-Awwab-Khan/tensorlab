import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorlab.linear_model import LogisticRegression  # Adjust import as needed

def test_simple_binary_classification():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.9

def test_zero_variation():
    X_zero_var = np.ones((10, 1))
    y_zero_var = np.zeros(10)
    model_zero_var = LogisticRegression(lr=0.01, n_iters=100)
    model_zero_var.fit(X_zero_var, y_zero_var)
    y_pred_zero_var = model_zero_var.predict(X_zero_var)
    assert np.allclose(y_pred_zero_var, y_zero_var)

def test_random_binary_classification_with_noise():
    np.random.seed(42)
    X_noise = np.random.rand(100, 2)
    y_noise = (X_noise[:, 0] * 0.3 + X_noise[:, 1] * 0.7 + np.random.randn(100) * 0.1 > 0.5).astype(int)
    X_train_noise, X_test_noise, y_train_noise, y_test_noise = train_test_split(X_noise, y_noise, test_size=0.2, random_state=42)
    model_noise = LogisticRegression(lr=0.05, n_iters=2000)
    model_noise.fit(X_train_noise, y_train_noise)
    y_pred_noise = model_noise.predict(X_test_noise)
    assert accuracy_score(y_test_noise, y_pred_noise) > 0.7
