import numpy as np
from typing import Tuple

class LassoRegression:
    def __init__(self, lr: float = 0.001, n_iters: int = 1000, alpha: float = 1.0):
        self.lr = lr
        self.alpha = alpha
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iters):
            y_hat = self._forward(X)
            dw, db = self._backward(X, y, y_hat)

            self.update_weights(dw, db)
            if _ % 100 == 0:
                print(f"Iteration {_} loss {self._loss(y, y_hat)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    def _backward(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, float]:
        m, n = X.shape
        dw = (1/m) * (X.T @ (y_hat - y)) + (self.alpha / m) * np.sum(np.sign(self.w))
        db = np.mean(y_hat - y)

        return dw, db

    def _loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        mse_loss = np.sum((y_hat - y) ** 2) / (2 * len(y))
        l1_regularization = self.alpha * np.sum(np.abs(self.w))
        return mse_loss + l1_regularization

    def update_weights(self, dw: float, db: float) -> None:
        self.w -= self.lr * dw
        self.b -= self.lr * db
