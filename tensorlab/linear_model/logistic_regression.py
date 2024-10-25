import numpy as np
from typing import Tuple

class LogisticRegression:
    def __init__(self, lr: float = 0.001, n_iters: int = 1000):
        """
        Logistic Regression model.
        Parameters:
        lr: float
            Learning rate
        n_iters: int
            Number of iterations
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Parameters:
        X: np.ndarray
            Features
        y: np.ndarray
            Target
        Returns:
        np.ndarray
            Predicted target
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(linear_model)
        return y_hat

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid function.
        Parameters:
        x: np.ndarray
            Input
        Returns:
        np.ndarray
            Sigmoid of input
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        Parameters:
        X: np.ndarray
            Features
        y: np.ndarray
            Target
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_hat = self._forward(X)
            dw, db = self._backward(X, y, y_hat)

            self.update_weights(dw, db)
            if _ % 100 == 0:
                print(f'Iteration: {_}, Loss: {self._loss(y, y_hat)}')

    def update_weights(self, dw: np.ndarray, db: float) -> None:
        """
        Update the weights and bias.
        Parameters:
        dw: np.ndarray
            Gradient of the weights
        db: float
            Gradient of the bias
        """
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def _loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Loss function.
        Parameters:
        y: np.ndarray
            Target
        y_hat: np.ndarray
            Predicted target
        Returns:
        float
            Loss
        """
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)  # Avoid log(0)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def _backward(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Backward pass.
        Parameters:
        X: np.ndarray
            Features
        y: np.ndarray
            Target
        y_hat: np.ndarray
            Predicted target
        Returns:
        np.ndarray
            Gradient of the weights
        """
        dw = np.dot(X.T, (y_hat - y)) / y.size
        db = np.sum(y_hat - y) / y.size
        return dw, db


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
        return np.round(self._forward(X))
