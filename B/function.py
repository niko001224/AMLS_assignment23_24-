import tensorflow as tf
from keras import layers, models
import numpy as np

def CNN_classification(X_train, y_train, X_val, y_val, width, height, channels, num_classes):

    model = models.Sequential([
    layers.Conv2D(32, 
                  (3, 3), 
                  activation='relu', 
                  input_shape=(width, height, channels)), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, 
                  (3,3), 
                  activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, 
                  (3,3), 
                  activation = 'relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), 
    layers.Dense(128, activation = 'relu'), 
    layers.Dense(num_classes, activation = 'softmax'),  
    ]) 

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(X_train, 
              y_train, 
              epochs=12, 
              batch_size=32, 
              validation_data=(X_val, y_val))

    test_loss, test_acc = model.evaluate(X_val, y_val)

    return test_acc, test_loss
