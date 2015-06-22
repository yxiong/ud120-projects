#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

classifier = svm.SVC(C=10000.0, kernel="rbf")
t0 = time()
classifier.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
predictions = classifier.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "accuracy score:", accuracy_score(predictions, labels_test)

print "Predictions for (10, 26, 50):", \
    predictions[10], predictions[26], predictions[50]
print "Number of emails by Chris:", np.sum(predictions == 1)


#########################################################


