import numpy as np
from A.function import svm_classification, RandomForest_classification, KNN_classification, LogisticRegression_classification
from A.DataReshape import datareshape
from B.function import CNN_classification
from keras.utils import to_categorical

#############    TaskA     ###############
# Use four machine learning methods to solve the problem: SVM, KNN, Logistic Regression, Random Forest
############     TaskA     ###############

# Load data
dataA = np.load('./Datasets/PneumoniaMNIST/pneumoniamnist.npz')
X_trainA = dataA['train_images']  
y_trainA = dataA['train_labels']
X_testA = dataA['test_images']
y_testA = dataA['test_labels']
X_valA = dataA['val_images']
y_valA = dataA['val_labels']

# Reshape data
X_trainAreshaped, y_trainAreshaped = datareshape(X_trainA, y_trainA)
X_testAreshaped, y_testAreshaped = datareshape(X_testA, y_testA)

# Method1: SVM
accuracy, report = svm_classification(X_trainAreshaped, y_trainAreshaped, X_testAreshaped, y_testAreshaped)
print(f"Accuracy Training of SVM: {accuracy * 100:.2f}%")
print("SVM Classification Report:")
print(report)

# Method2: Random Forest
accuracy, report = RandomForest_classification(X_trainAreshaped, y_trainAreshaped, X_testAreshaped, y_testAreshaped)
print(f"Accuracy Training of Random Forest: {accuracy * 100:.2f}%")
print("Random Forest Classification Report:")
print(report)

# Method3: Logistic Regression
accuracy, report = LogisticRegression_classification(X_trainAreshaped, y_trainAreshaped, X_testAreshaped, y_testAreshaped)
print(f"Accuracy Training of Logistic Regression: {accuracy * 100:.2f}%")
print("Logisitic Regression Classification Report:")
print(report)

# Method4: KNN
accuracy, report = KNN_classification(X_trainAreshaped, y_trainAreshaped, X_testAreshaped, y_testAreshaped)
print(f"Accuracy Training of KNN: {accuracy * 100:.2f}%")
print("KNN Classification Report:")
print(report)

#############    TaskB     ###############
# Use CNN to solve the task
############     TaskB     ###############

# Load data
dataB = np.load('./Datasets/PathMNIST/pathmnist.npz')
X_trainB = dataB['train_images']  
y_trainB = dataB['train_labels']
X_testB = dataB['test_images']
y_testB = dataB['test_labels']
X_valB = dataB['val_images']
y_valB = dataB['val_labels']

width, height, channels = X_trainB.shape[1], X_trainB.shape[2], X_trainB.shape[3]
num_classes = 9
y_trainB = to_categorical(y_trainB, num_classes)
y_testB = to_categorical(y_testB, num_classes)

# Method: CNN
test_acc, test_loss = CNN_classification(X_trainB, y_trainB, X_testB, y_testB, width, height, channels, num_classes)
print(f"Accuracy Training of CNN: {test_acc * 100:.2f}%")
