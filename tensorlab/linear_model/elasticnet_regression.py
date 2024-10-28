import numpy as np
from typing import Tuple

class ElasticNetRegression:
    def __init__(self, lr: float = 0.001, alpha: float = 0.1, rho: float = 0.5, n_iters: int = 1000):
        self.alpha = alpha
        self.lr = lr
        self.rho = rho
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
                print(f'Iteration: {_}, Loss: {self._loss(y, y_hat)}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    def _backward(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, float]:
        m, n = X.shape
        dw = (1/m) * np.dot(X.T, (y_hat - y)) + self.alpha * (self.rho * np.sign(self.w) + (1 - self.rho) * self.w)
        db = (1/m) * np.sum(y_hat - y)
        return dw, db

    def update_weights(self, dw: np.ndarray, db: float) -> None:
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def _loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.mean((y - y_hat) ** 2) + self.alpha * (self.rho * np.sum(np.abs(self.w)) + 0.5 * (1 - self.rho) * np.sum(self.w ** 2))
