import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
param_grid = {'n_neighbors': [3, 4, 5, 7, 8]}

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