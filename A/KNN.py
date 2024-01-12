import numpy as np
from DataReshape import datareshape
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from DataReshape import datareshape

# Load data
data = np.load('./Datasets/PneumoniaMNIST/pneumoniamnist.npz')
X_train = data['train_images']  
y_train = data['train_labels']
X_val = data['val_images']
y_val = data['val_labels']

# Reshape data
X_train, y_train = datareshape(X_train, y_train)
X_val, y_val = datareshape(X_val, y_val)

# Choose the n_neighbor from 3, 5, 7 ,9 
param_grid = {'n_neighbors': [3, 5, 7, 9]}

# Create KNN classifier
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_n_neighbors = grid_search.best_params_['n_neighbors']

# Create KNN classifier with the best parameter
best_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Best n_neighbors: {best_n_neighbors}")
print(f"Accuracy with best parameter: {accuracy * 100:.2f}%")
# Best n_neighbour = 9
print('Classification report：\n', classification_report(y_val, y_pred))
