import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso

class PolynomialRegressionWrapper:
    def __init__(self, degree=2, regularization=None, alpha=1.0):
        self.degree = degree
        self.regularization = regularization
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=self.degree)
        if regularization == 'l2':
            self.model = Ridge(alpha=self.alpha)
        elif regularization == 'l1':
            self.model = Lasso(alpha=self.alpha)
        else:
            self.model = None

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X.cpu().numpy())
        y = y.cpu().numpy()
        if self.model:
            self.model.fit(X_poly, y)
        else:
            self.coefficients = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        X_poly = self.poly.transform(X.cpu().numpy())
        if self.model:
            return self.model.predict(X_poly)
        else:
            return X_poly @ self.coefficients
