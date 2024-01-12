import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


data = np.load('./Datasets/PathMNIST/pathmnist.npz')
X_train = data['train_images']  
y_train = data['train_labels']
X_val = data['val_images']
y_val = data['val_labels']

width, height, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
num_classes = 9
y_train_one_hot = to_categorical(y_train, num_classes)
y_val_one_hot = to_categorical(y_val, num_classes)

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
model.summary()
  
 
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Train model
history = model.fit(X_train, y_train_one_hot, 
                 epochs=20, 
                 batch_size=16, 
                 shuffle = True,
                 validation_data=(X_val, y_val_one_hot),
                 verbose = 2)

test_loss, test_acc = model.evaluate(X_val, y_val_one_hot)
print(f'Test accuracy: {test_acc * 100:.2f}%')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs,acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
