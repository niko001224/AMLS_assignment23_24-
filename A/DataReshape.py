import numpy as np
from sklearn.decomposition import PCA

def datareshape(X, y):
    # Reshape X from 3D to 2D, y from 2D to 1D
    num_samples, height, width = X.shape
    X_reshaped = X.reshape(num_samples, height * width)
    y_reshaped = y.ravel()

    return X_reshaped, y_reshaped

def datareshape_PCA(X, y):

    # Reshape X from 3D to 2D, y from 2D to 1D
    X_flatten = X.reshape((X.shape[0], -1))
    pca = PCA(n_components=2)  
    X_reshaped = pca.fit_transform(X_flatten)
    y_reshaped = y.ravel()
    
    return X_reshaped, y_reshaped

