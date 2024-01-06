from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

#Train and predict with the method of SVM
def svm_classification(X_train, y_train, X_test, y_test):
   
    clf = svm.SVC(kernel='rbf')# kernel: linear, rbf, poly, sigmoid
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    #Calculate accurracy
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report