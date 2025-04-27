import numpy as np

def gaussian_rbf_kernel(X, Y, sigma):
    """
    Gaussian Radial Basis Function Kernel
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    distances = XX + YY - 2 * np.dot(X, Y.T)
    return np.exp(-distances / (2 * sigma**2))



def linear_kernel(X, Y):
    """
    Linear Kernel
    k(x, y) = x.T @ y
    """
    return np.dot(X, Y.T)

def polynomial_kernel(X, Y, degree=3, coef0=1):
    """
    Polynomial Kernel
    k(x, y) = (x.T @ y + coef0)^degree
    """
    return (np.dot(X, Y.T) + coef0) ** degree

def laplace_kernel(X, Y, sigma):
    """
    Laplace Kernel
    k(x, y) = exp(-||x - y||_1 / sigma)
    """
    distances = np.sum(np.abs(X[:, None] - Y), axis=2)  # L1 norm
    return np.exp(-distances / sigma)

def sigmoid_kernel(X, Y, alpha=1.0, coef0=0.0):
    """
    Sigmoid Kernel
    k(x, y) = tanh(alpha * (x.T @ y) + coef0)
    """
    return np.tanh(alpha * np.dot(X, Y.T) + coef0)

def cosine_kernel(X, Y):
    """
    Cosine Kernel
    k(x, y) = (x.T @ y) / (||x|| * ||y||)
    """
    norm_X = np.linalg.norm(X, axis=1)[:, None]
    norm_Y = np.linalg.norm(Y, axis=1)[None, :]
    return np.dot(X, Y.T) / (norm_X * norm_Y + 1e-10)  # Avoid division by zero


def exponential_kernel(X, Y, sigma):
    """
    Exponential Kernel
    k(x, y) = exp(-||x - y|| / sigma)
    """
    distances = np.sum(np.abs(X[:, None] - Y), axis=2)  # L1 norm
    return np.exp(-distances / sigma)

def anova_kernel(X, Y, sigma, d=2):
    """
    ANOVA Kernel
    k(x, y) = sum_{i=1}^n exp(-sigma * (x_i - y_i)^2)^d
    """
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[1]):
        K += np.exp(-sigma * (X[:, i][:, None] - Y[:, i])**2)**d
    return K