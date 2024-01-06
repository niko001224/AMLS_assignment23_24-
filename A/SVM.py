import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from DataReshape import datareshape
from sklearn.model_selection import GridSearchCV

# Load data
data = np.load('./Datasets/PneumoniaMNIST/pneumoniamnist.npz')
X_train = data['train_images']  
y_train = data['train_labels']
X_test = data['test_images']
y_test = data['test_labels']
X_val = data['val_images']
y_val = data['val_labels']

# Reshape the data from 3D to 2D
X_trainreshaped, y_trainreshaped = datareshape(X_train, y_train)
X_valreshaped, y_valreshaped = datareshape(X_val, y_val)
X_testreshaped, y_testreshaped = datareshape(X_test, y_test)

# Create SVM classifier (kernal choose from: linear, rbf, poly, sigmoid )
clf = svm.SVC(kernel='rbf')

# Train the model
clf.fit(X_trainreshaped, y_trainreshaped)

# Make predictions (use validation images)
y_pred = clf.predict(X_valreshaped)

# Calculate accuracy (use validation images)
accuracy = accuracy_score(y_valreshaped, y_pred)
print(f"Accuracy Training: {accuracy * 100:.2f}%")
report = classification_report(y_valreshaped, y_pred)
print("SVM Classification Report:")
print(report)

# Finding the best C & gamma value
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_trainreshaped, y_trainreshaped)

best_svm_model = grid_search.best_estimator_
y_test_pred = best_svm_model.predict(X_testreshaped)

final_accuracy = accuracy_score(y_testreshaped, y_test_pred)
print(f"Final Accuracy: {final_accuracy * 100:.2f}%")

