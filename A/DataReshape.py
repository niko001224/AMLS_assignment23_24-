import numpy as np

def datareshape(X, y):
    # Reshape X from 3D to 2D, y from 2D to 1D
    num_samples, height, width = X.shape
    X_reshaped = X.reshape(num_samples, height * width)
    y_reshaped = y.ravel()

    return X_reshaped, y_reshaped
