import numpy as np
import pytest
from tensorlab.linear_model import LassoRegression

@pytest.fixture
def simple_data():
    # Fixture to provide a simple dataset
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    return X, y

@pytest.fixture
def lasso_model():
    # Fixture to create a LassoRegression instance
    return LassoRegression(lr=0.01, n_iters=1000, alpha=0.1)

def test_initial_weights_bias_none(lasso_model):
    # Test initial weights and bias are None before fitting
    assert lasso_model.w is None, "Weights should be None before fitting"
    assert lasso_model.b is None, "Bias should be None before fitting"

def test_fit_initializes_weights_bias(lasso_model, simple_data):
    # Test weights and bias are initialized after fitting
    X, y = simple_data
    lasso_model.fit(X, y)
    assert lasso_model.w is not None, "Weights should be initialized after fitting"
    assert lasso_model.b is not None, "Bias should be initialized after fitting"

def test_predict_output_shape(lasso_model, simple_data):
    # Test predict output shape matches input shape
    X, y = simple_data
    lasso_model.fit(X, y)
    predictions = lasso_model.predict(X)
    assert predictions.shape == y.shape, "Predictions shape should match target shape"

def test_fit_predict_simple_data(lasso_model, simple_data):
    # Test model fitting and predicting on simple data
    X, y = simple_data
    lasso_model.fit(X, y)
    predictions = lasso_model.predict(X)
    # Check that predictions are close to actual y (small dataset, should fit well)
    assert np.allclose(predictions, y, atol=0.5), "Predictions should be close to the true values"

def test_lasso_regularization_effect(simple_data):
    # Test Lasso regularization effect by comparing with and without regularization
    X, y = simple_data

    # Model without regularization
    model_no_reg = LassoRegression(lr=0.01, n_iters=1000, alpha=0.0)
    model_no_reg.fit(X, y)
    preds_no_reg = model_no_reg.predict(X)

    # Model with regularization
    model_with_reg = LassoRegression(lr=0.01, n_iters=1000, alpha=0.5)
    model_with_reg.fit(X, y)
    preds_with_reg = model_with_reg.predict(X)

    # Check that regularization makes coefficients smaller (absolute values decrease)
    assert np.sum(np.abs(model_with_reg.w)) < np.sum(np.abs(model_no_reg.w)), "Lasso should reduce weight magnitudes"

def test_loss_decreases_during_training(lasso_model, simple_data):
    # Test that loss decreases during training
    X, y = simple_data
    losses = []
    for _ in range(lasso_model.n_iters):
        y_hat = lasso_model._forward(X)
        losses.append(lasso_model._loss(y, y_hat))
        dw, db = lasso_model._backward(X, y, y_hat)
        lasso_model.update_weights(dw, db)

    # Check if losses are decreasing
    assert all(x >= y for x, y in zip(losses, losses[1:])), "Loss should decrease with training"

def test_zero_regularization_equivalent_to_linear(simple_data):
    # Test if Lasso with zero alpha behaves like standard linear regression
    X, y = simple_data
    model_zero_reg = LassoRegression(lr=0.01, n_iters=1000, alpha=0.0)
    model_zero_reg.fit(X, y)
    predictions = model_zero_reg.predict(X)

    # Check if predictions are close to y
    assert np.allclose(predictions, y, atol=0.5), "Lasso with zero regularization should behave like linear regression"
