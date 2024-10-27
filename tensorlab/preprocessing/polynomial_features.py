import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, deg: int = 2):
        """
        Parameters
        ----------
        deg : int, default=2
            The degree of polynomial features.
        """
        self.deg = deg

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PolynomialFeatures to the data.
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.
        """
        self.n_features = X.shape[1]

        self.combinations = [
            comb for deg in range(1, self.deg + 1)
            for comb in combinations_with_replacement(range(self.n_features), deg)
        ]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data to polynomial features.
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        expanded_features : np.ndarray of shape (n_samples, n_output_features)
            The polynomial features.
        """
        n_samples = X.shape[0]

        expanded_features = np.ones((n_samples, 1))

        for comb in self.combinations:
            expanded_features = np.hstack([
                expanded_features, np.prod(X[:, comb], axis=1, keepdims=True)
            ])

        return expanded_features

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PolynomialFeatures to the data and transform the input data to polynomial features.
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        expanded_features : np.ndarray of shape (n_samples, n_output_features)
            The polynomial features.
        """
        self.fit(X)
        return self.transform(X)
