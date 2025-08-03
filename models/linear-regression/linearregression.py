import numpy as np

class LinearRegression:
    def __init__(self, degree=1, learning_rate=0.01, num_iterations=1000, lambda_=0.0, regularization=None):
        """
        degree: Degree of the polynomial features.
        learning_rate: Learning rate for gradient descent.
        num_iterations: Number of iterations for gradient descent.
        lambda_: Regularization strength.
        regularization: Type of regularization ('l1' for Lasso, 'l2' for Ridge, None for no regularization).
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.regularization = regularization
        self.theta_ = None

    def _polynomial_features(self, X):
        """Generate polynomial features up to the specified degree."""
        X_poly = np.hstack([X ** i for i in range(self.degree + 1)])
        return X_poly

    def fit(self, X, y):
        """Fit the model to the data using gradient descent."""
        X_poly = self._polynomial_features(X)
        m, n = X_poly.shape
        self.theta_ = np.zeros(n)
        
        for _ in range(self.num_iterations):
            predictions = X_poly @ self.theta_
            errors = predictions - y
            
            # Compute gradient with regularization
            if self.regularization == 'l2':
                gradients = (X_poly.T @ errors + self.lambda_ * self.theta_) / m
                gradients[0] = (X_poly[:, 0].T @ errors) / m  # Don't regularize the intercept
            elif self.regularization == 'l1':
                gradients = (X_poly.T @ errors + self.lambda_ * np.sign(self.theta_)) / m
                gradients[0] = (X_poly[:, 0].T @ errors) / m  # Don't regularize the intercept
            else:
                gradients = (X_poly.T @ errors) / m

            # Update theta
            self.theta_ -= self.learning_rate * gradients

    def predict(self, X):
        """Predict the output for the given input data."""
        X_poly = self._polynomial_features(X)
        return X_poly @ self.theta_

    def mse(self, y_true, y_pred):
        """Calculate Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    def std_deviation(self, y_true, y_pred):
        """Calculate standard deviation of the errors."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def variance(self, y_true, y_pred):
        """Calculate variance of the errors."""
        return np.var(y_true - y_pred)
