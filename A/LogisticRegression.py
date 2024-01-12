import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from DataReshape import datareshape

# Load data
data = np.load('./Datasets/PneumoniaMNIST/pneumoniamnist.npz')
X_train = data['train_images']  
y_train = data['train_labels']
X_val = data['val_images']
y_val = data['val_labels']

# Reshape the data 
X_train, y_train = datareshape(X_train, y_train)
X_val, y_val = datareshape(X_val, y_val)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print('Classification report：\n', classification_report(y_val, y_pred))