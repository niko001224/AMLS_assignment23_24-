import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
from DataReshape import datareshape

# Load data
data = np.load('./Datasets/PneumoniaMNIST/pneumoniamnist.npz')
X_train = data['train_images']  
y_train = data['train_labels']
X_test = data['test_images']
y_test = data['test_labels']
X_val = data['val_images']
y_val = data['val_labels']

# Reshape the data 
X_train, y_train = datareshape(X_train, y_train)
X_val, y_val = datareshape(X_val, y_val)
X_test, y_test = datareshape(X_test, y_test)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)  
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print('Classification report：\n', classification_report(y_val, y_pred))

# Drawing decision tree
tree_params={
    'criterion':'entropy'
}
clf = tree.DecisionTreeClassifier( **tree_params )
clf.fit(X_train,y_train)

feature_names = [f"Image_{i}" for i in range(X_train.shape[0])]
class_names = ['Normal', 'Pneumonia']  

#Function to visualise the decision tree   
def visualise_tree(tree_to_print):
    plt.figure()
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
    tree.plot_tree(tree_to_print,
               feature_names = feature_names,
               class_names = class_names, 
               filled = True,
              rounded=True);
    plt.show()


visualise_tree(clf)


