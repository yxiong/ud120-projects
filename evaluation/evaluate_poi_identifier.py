#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size = 0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
print "test accuracy:", accuracy_score(predictions, labels_test)

print "# of POIs predicted in test set:", np.sum(predictions)
print "# of people in test set:", len(predictions)
print "accuracy if all non-POI:", accuracy_score(np.zeros(len(predictions)),
                                                 labels_test)
print "# of true positives:", np.sum(np.logical_and(predictions, labels_test))
print "precision score:", precision_score(labels_test, predictions)
print "recall score:", recall_score(labels_test, predictions)

print "################################################################"
p = np.array([0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
t = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
print "# of true positives:", np.sum(np.logical_and(p, t))
print "# of true negatives:", np.sum(
    np.logical_and(np.logical_not(p), np.logical_not(t)))
print "# of false positives:", np.sum(np.logical_and(p, np.logical_not(t)))
print "# of false negatives:", np.sum(np.logical_and(np.logical_not(p), t))
print "precision score:", precision_score(t, p)
print "recall score:", recall_score(t, p)
