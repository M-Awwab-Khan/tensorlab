import numpy as np
import pytest
from tensorlab.svm import LinearSVC  # Replace 'your_module' with the actual module name where LinearSVC is defined.

@pytest.fixture
def data():
    # Sample data for testing
    X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, -1, -1])
    return X, y

def test_fit_shape(data):
    X, y = data
    model = LinearSVC(lr=0.001, n_iter=10, C=1.0)
    model.fit(X, y)
    assert model.w.shape == (X.shape[1],), "Weights shape does not match number of features."
    assert isinstance(model.b, float), "Bias should be a float."

def test_predict(data):
    X, y = data
    model = LinearSVC(lr=0.001, n_iter=100, C=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape, "Prediction shape should match target shape."
    assert np.all(np.isin(predictions, [-1, 1])), "Predictions should only contain -1 or 1."

def test_margin_condition(data):
    X, y = data
    model = LinearSVC(lr=0.001, n_iter=500, C=1.0)
    model.fit(X, y)
    # Checking if each sample satisfies the margin condition
    margin_condition = y * (np.dot(X, model.w) + model.b) >= 1 - 0.1
    print(model.w, model.b)
    assert np.all(margin_condition), "All points should satisfy the margin condition after training."

def test_loss_decrease(data):
    X, y = data
    model = LinearSVC(lr=0.001, n_iter=200, C=1.0)
    initial_loss = model._loss(X, y)
    model.fit(X, y)
    final_loss = model._loss(X, y)
    assert final_loss <= initial_loss, "Loss should decrease or stay constant after training."

def test_hyperparameter_effect(data):
    X, y = data
    model_high_C = LinearSVC(lr=0.001, n_iter=100, C=10.0)
    model_low_C = LinearSVC(lr=0.001, n_iter=100, C=0.1)
    model_high_C.fit(X, y)
    model_low_C.fit(X, y)
    # With higher C, expect fewer margin violations (harder margin)
    violations_high_C = np.sum(y * (np.dot(X, model_high_C.w) - model_high_C.b) < 1)
    violations_low_C = np.sum(y * (np.dot(X, model_low_C.w) - model_low_C.b) < 1)
    assert violations_high_C <= violations_low_C, "Higher C should result in fewer margin violations."
