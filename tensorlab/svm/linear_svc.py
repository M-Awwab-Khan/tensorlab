import numpy as np

class LinearSVC:
    def __init__(self, lr: float = 0.001, n_iter: int = 1000, C: float = 1.0):
        """
        Support Vector Classifier.
        Parameters:
        lr: float - Learning rate.
        n_iters: int - Number of iterations.
        C: float - Regularization parameter.
        """
        self.lr = lr
        self.n_iter = n_iter
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVM model to the data.
        Parameters:
        X: np.ndarray - Features, shape (n_samples, n_features).
        y: np.ndarray - Target labels (-1 or 1), shape (n_samples,).
        """

        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        y = np.where(y <= 0, -1, 1)  # Ensure labels are -1 and 1

        for i in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    dw = self.w
                    db = 0
                else:
                    dw = self.w - self.C * y[idx] * x_i
                    db = -self.C * y[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db


            if i % 100 == 0:
                print(f'Iteration: {i}, Loss: {self._loss(X, y)}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target labels.
        Parameters:
        X: np.ndarray - Features, shape (n_samples, n_features).
        Returns:
        np.ndarray - Predicted target labels, shape (n_samples,).
        """
        return np.sign(np.dot(X, self.w) - self.b)


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss function.
        Parameters:
        X: np.ndarray - Features, shape (n_samples, n_features).
        y: np.ndarray - Target labels (-1 or 1), shape (n_samples,).
        Returns:
        float - Loss value.
        """
        hinge_loss = 1 - y * (np.dot(X, self.w) - self.b)
        hinge_loss = np.maximum(0, hinge_loss)
        return 0.5 * np.dot(self.w, self.w) + self.C * np.mean(hinge_loss)
