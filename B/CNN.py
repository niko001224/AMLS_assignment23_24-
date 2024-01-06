import tensorflow as tf
from keras import layers, models
import numpy as np
from keras.utils import to_categorical

data = np.load('C:/Users/32216/Desktop/AMLS/AMLS_23-24_SN12345678/AMLS_assignment23_24-/Datasets/PathMNIST/pathmnist.npz')
X_train = data['train_images']  
y_train = data['train_labels']
X_val = data['val_images']
y_val = data['val_labels']

width, height, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
num_classes = 9
y_train_one_hot = to_categorical(y_train, num_classes)
y_val_one_hot = to_categorical(y_val, num_classes)

# CNN Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_val, y_val_one_hot))

# Calculate accuracy
test_loss, test_acc = model.evaluate(X_val, y_val_one_hot)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# Make prediction
predictions = model.predict(X_val)
