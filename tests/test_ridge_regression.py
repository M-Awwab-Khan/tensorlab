import numpy as np
import pytest
from tensorlab.linear_model import RidgeRegression

@pytest.fixture
def sample_data():
    # Sample linear dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([5, 7, 9, 11, 13])
    return X, y

def test_ridge_fit(sample_data):
    X, y = sample_data
    model = RidgeRegression(alpha=0.1, lr=0.01, n_iter=1000)
    model.fit(X, y)
    assert model.w is not None, "Model weights should not be None after fitting"
    assert model.bias is not None, "Model bias should not be None after fitting"

def test_ridge_predict(sample_data):
    X, y = sample_data
    model = RidgeRegression(alpha=0.1, lr=0.01, n_iter=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape, "Predictions shape should match the target shape"
    # Check if predictions are close to expected values (since it's a simple dataset)
    assert np.allclose(predictions, y, atol=1.0), "Predictions should be close to actual values within tolerance"

def test_ridge_regularization_effect():
    X = np.random.rand(100, 3)
    y = 5 * X[:, 0] + 3 * X[:, 1] + 1.5 * X[:, 2] + np.random.normal(0, 0.1, 100)

    # High regularization
    model_high_alpha = RidgeRegression(alpha=10.0, lr=0.01, n_iter=1000)
    model_high_alpha.fit(X, y)
    weights_high_alpha = np.sum(np.abs(model_high_alpha.w))

    # Low regularization
    model_low_alpha = RidgeRegression(alpha=0.01, lr=0.01, n_iter=1000)
    model_low_alpha.fit(X, y)
    weights_low_alpha = np.sum(np.abs(model_low_alpha.w))

    assert weights_high_alpha < weights_low_alpha, "High regularization should result in smaller weight values"

def test_ridge_loss_reduction(sample_data):
    X, y = sample_data
    model = RidgeRegression(alpha=0.1, lr=0.01, n_iter=1000)
    initial_loss = model._loss(y, model._forward(X))
    model.fit(X, y)
    final_loss = model._loss(y, model._forward(X))
    assert final_loss < initial_loss, "Loss should decrease after training"
