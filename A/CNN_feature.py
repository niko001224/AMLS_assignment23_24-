import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

# 假设你的数据是一个NumPy数组，形状为 (样本数, 高度, 宽度, 通道数)
# 请根据实际情况调整数据加载方式
data = np.load('./Datasets/PneumoniaMNIST/pneumoniamnist.npz')
X_train = data['train_images']  
y_train = data['train_labels']
X_test = data['test_images']
y_test = data['test_labels']
X_val = data['val_images']
y_val = data['val_labels']

width, height = X_train.shape[1], X_train.shape[2]
channels = 1

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练CNN模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 提取CNN模型的倒数第二层输出作为特征
cnn_features_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)
X_train_features = cnn_features_model.predict(X_train)
X_test_features = cnn_features_model.predict(X_test)

# 使用SVM进行分类
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_features, y_train)

# 评估性能
y_pred = svm_model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
