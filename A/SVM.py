import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from DataReshape import datareshape
from sklearn.model_selection import GridSearchCV
#from CNN_feature import CNN_feature_aquire

# Load data
data = np.load('./Datasets/PneumoniaMNIST/pneumoniamnist.npz')
X_train = data['train_images']  
y_train = data['train_labels']
X_test = data['test_images']
y_test = data['test_labels']
X_val = data['val_images']
y_val = data['val_labels']

# Reshape the data
X_trainreshaped, y_trainreshaped = datareshape(X_train, y_train)
X_valreshaped, y_valreshaped = datareshape(X_val, y_val)
# Feature of the data
#X_trainfeature, y_trainfeature = CNN_feature_aquire(X_train, y_train)


# Create SVM classifier (kernal choose from: linear, rbf, poly, sigmoid )
clf = svm.SVC(kernel='rbf')
clf.fit(X_trainreshaped, y_trainreshaped)
y_pred = clf.predict(X_valreshaped)

# Calculate accuracy (use validation images)
accuracy = accuracy_score(y_valreshaped, y_pred)
print(f"Accuracy Training: {accuracy * 100:.2f}%")
report = classification_report(y_valreshaped, y_pred)
print("SVM Classification Report:")
print(report)

# Finding the best C & gamma value
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_trainreshaped, y_trainreshaped)

best_svm_model = grid_search.best_estimator_
y_best_C = grid_search.best_params_['C']

print("Best C for the model:")
print(y_best_C)


y_val_pred = best_svm_model.predict(X_valreshaped)
final_accuracy = accuracy_score(y_valreshaped, y_val_pred)
print(f"Final Accuracy: {final_accuracy * 100:.2f}%")
report = classification_report(y_valreshaped, y_val_pred)
print("SVM Classification Report:")
print(report)
