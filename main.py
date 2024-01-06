import numpy as np
from A.function import svm_classification
from A.DataReshape import datareshape

#############    TaskA     ###############
# Use four machine learning methods to solve the problem: SVM, KNN, Logistic Regression, Random Forest
# As the data of the images are all 3D, reshape the data from 3D to 2D for the classification
# 
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



