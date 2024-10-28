import numpy as np
from typing import Tuple

class RidgeRegression:
    def __init__(self, alpha: float = 1.0, lr: float = 0.01, n_iter: int = 1000):
        """
        Ridge Regression model.
        Parameters:
        alpha: float
            Regularization parameter
        lr: float
            Learning rate
        n_iter: int
            Number of iterations
        """
        self.alpha = alpha
        self.w = None
        self.bias = None
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        Parameters:
        X: np.ndarray
            Features
        y: np.ndarray
            Target
        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iter):
            y_hat = self._forward(X)
            dw, db = self._backward(X, y, y_hat)

            self.update_weights(dw, db)
            if _ % 100 == 0:
                print(f'Iteration: {_}, Loss: {self._loss(y, y_hat)}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target.
        Parameters:
        X: np.ndarray
            Features
        Returns:
        np.ndarray
            Predicted target
        """

        return self._forward(X)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.bias is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before prediction.")
        return np.dot(X, self.w) + self.bias

    def _backward(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, float]:
        m, n = X.shape

        dw = (np.dot(X.T, y_hat - y) + self.alpha * self.w) / m
        db = np.sum(y_hat - y) / m

        return dw, db

    def update_weights(self, dw: np.ndarray, db: float) -> None:
        self.w -= self.lr * dw
        self.bias -= self.lr * db

    def _loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        m = len(y)
        return (np.sum((y - y_hat) ** 2) / (2 * m)) + (self.alpha * np.sum(self.w ** 2) / (2 * m))
