# AMLS_assignment23_24-
Assignment for AMLS

The project contains two main tasks:
A: Binary classification task (using PneumoniaMNIST dataset). The objective is to classify an image onto "Normal" (no pneumonia) or "Pneumonia" (presence of pneumonia)
B: Multi-class classification task (using PathMNIST dataset). The objective is to classify an image onto 9 different types of tissues. 

1 Organization of the project:
The main function runs the accuracy results for all model test set data, with folders A and B containing the pre-trained models and the modules to be implemented in the main function, respectively.

2 Role of each file
main.py : Accuracy test results for all classification models on the dataset can be obtained by running main.py

Folder A : SVM.py, RandomForest.py, LogisticRegression.py, KNN.py are the pre-trained models using training and 
           validation sets.
           function.py includes the functions of taskA to be run in the main function
           Datareshape.py is the function to reshape the data to be used in certain statements in the pre-trained models

Folder B : CNN.py is the pre-trained models using training and validation sets.
            function.py includes the functions of taskB to be run in the main function.

Folder Datasets: Folder PathMNIST includes data of TaskB with name 'pathmnist.npz'
                 Folder PneumoniaMNIST includes data of TaskB with name 'pneumoniamnist.npz'

3  Packages required to run the code
numpy
tensorflow
keras
sklearn
matplotlib
          








