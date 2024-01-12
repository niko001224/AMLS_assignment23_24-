import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def CNN_feature_aquire(X, y):

    width, height = X.shape[1], X.shape[2]
    channels = 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3,3), activation = 'relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), 
    layers.Dense(64, activation = 'relu'), 
    layers.Dense(1, activation = 'softmax'), 
    ]) 

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(X_train, 
              y_train, 
              epochs=7, 
              batch_size=16, 
              validation_data=(X_test, y_test))
    
    cnn_features_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)
    X_train_features = cnn_features_model.predict(X)
    
    return X_train_features
    

