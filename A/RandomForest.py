import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Create Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators as needed

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
