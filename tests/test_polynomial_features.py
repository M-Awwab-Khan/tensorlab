import numpy as np
import pytest
from tensorlab.preprocessing import PolynomialFeatures

def test_bias_term():
    X = np.array([[2], [3]])
    poly = PolynomialFeatures(deg=1)
    X_poly = poly.fit_transform(X)
    assert np.array_equal(X_poly, np.array([[1, 2], [1, 3]])), "Bias term not added correctly"

def test_degree_two():
    X = np.array([[2], [3]])
    poly = PolynomialFeatures(deg=2)
    X_poly = poly.fit_transform(X)
    # Expected [[1, x, x^2]] for each sample
    expected = np.array([[1, 2, 4], [1, 3, 9]])
    assert np.array_equal(X_poly, expected), "Degree 2 polynomial features incorrect"

def test_multifeature_degree_two():
    X = np.array([[1, 2], [3, 4]])
    poly = PolynomialFeatures(deg=2)
    X_poly = poly.fit_transform(X)
    # Expected terms: [1, x1, x2, x1^2, x1*x2, x2^2]
    expected = np.array([
        [1, 1, 2, 1, 2, 4],
        [1, 3, 4, 9, 12, 16]
    ])
    assert np.array_equal(X_poly, expected), "Multi-feature degree 2 polynomial features incorrect"

def test_degree_three():
    X = np.array([[2], [3]])
    poly = PolynomialFeatures(deg=3)
    X_poly = poly.fit_transform(X)
    # Expected [[1, x, x^2, x^3]] for each sample
    expected = np.array([[1, 2, 4, 8], [1, 3, 9, 27]])
    assert np.array_equal(X_poly, expected), "Degree 3 polynomial features incorrect"

def test_fit_transform_vs_transform():
    X = np.array([[2], [3]])
    poly = PolynomialFeatures(deg=2)
    X_poly_fit_transform = poly.fit_transform(X)
    poly.fit(X)
    X_poly_transform = poly.transform(X)
    assert np.array_equal(X_poly_fit_transform, X_poly_transform), "fit_transform and transform do not match"
