import numpy as np
import pytest
from tensorlab.linear_model import ElasticNetRegression  # Ensure your module is named correctly

# Test initialization
def test_initialization():
    model = ElasticNetRegression(alpha=1.0, rho=0.5, lr=0.01, n_iters=1000)
    assert model.alpha == 1.0
    assert model.rho == 0.5
    assert model.lr == 0.01
    assert model.n_iters == 1000
    assert model.w is None
    assert model.b is None

# Test fitting on a small dataset
def test_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = ElasticNetRegression(alpha=1.0, rho=0.5, lr=0.01, n_iters=1000)
    model.fit(X, y)

    assert model.w is not None, "Weights should not be None after fitting"
    assert model.b is not None, "Bias should not be None after fitting"

# Test predictions
def test_predict():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = ElasticNetRegression(alpha=0.1, rho=0.5, lr=0.01, n_iters=1000)
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == y.shape, "Prediction shape mismatch"
    assert np.allclose(predictions, y, atol=1), "Predictions should be close to actual values"

# Test regularization effect
def test_regularization_effect():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])

    # Strong regularization should drive some weights toward zero
    model = ElasticNetRegression(alpha=10.0, rho=0.9, lr=0.01, n_iters=1000)
    model.fit(X, y)
    assert np.all(np.abs(model.w) < 1), "Weights should be reduced with strong regularization"

# Test learning rate effect on convergence
def test_learning_rate():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])

    # Lower learning rate, slower convergence
    model_slow = ElasticNetRegression(alpha=1.0, rho=0.5, lr=0.0001, n_iters=1000)
    model_slow.fit(X, y)
    loss_slow = np.mean((model_slow.predict(X) - y) ** 2)

    # Higher learning rate, faster convergence
    model_fast = ElasticNetRegression(alpha=1.0, rho=0.5, lr=0.01, n_iters=1000)
    model_fast.fit(X, y)
    loss_fast = np.mean((model_fast.predict(X) - y) ** 2)

    assert loss_fast < loss_slow, "Higher learning rate should yield lower loss in the same number of iterations"

# Test if ElasticNet handles zero variance in features
def test_zero_variance_feature():
    X = np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]])
    y = np.array([2, 3, 4, 5])

    model = ElasticNetRegression(alpha=1.0, rho=0.5, lr=0.01, n_iters=1000)
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == y.shape, "Prediction shape mismatch"
    assert np.allclose(predictions, y, atol=1), "Predictions should be close to actual values despite zero variance feature"
