import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data
data = np.load('C:/Users/32216/Desktop/AMLS/AMLS_23-24_SN12345678/AMLS_assignment23_24-/Datasets/PneumoniaMNIST/pneumoniamnist.npz')
X = data['train_images']  
y = data['train_labels']

# Reshape the data from 3D to 2D
num_samples, height, width = X.shape
X_reshaped = X.reshape(num_samples, height * width)
y = y.ravel()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=42)

# Define a range of values for n_neighbors
param_grid = {'n_neighbors': [3, 5, 7, 9]}

# Create KNN classifier
knn = KNeighborsClassifier()

# Create GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameter
best_n_neighbors = grid_search.best_params_['n_neighbors']

# Create KNN classifier with the best parameter
best_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)

# Train the model with the best parameter
best_knn.fit(X_train, y_train)

# Make predictions
y_pred = best_knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Best n_neighbors: {best_n_neighbors}")
print(f"Accuracy with best parameter: {accuracy * 100:.2f}%")
# Best n_neighbour = 7
print('Classification report：\n', classification_report(y_test, y_pred))

X_flatten = X_train.reshape((X_train.shape[0], -1))
# 可视化决策边界
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = best_knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolor='k')
plt.title("KNN Binary Classification")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()