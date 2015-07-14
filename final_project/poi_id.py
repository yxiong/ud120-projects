#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'restricted_stock',
                 'exercised_stock_options', 'expenses']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
del data_dict["TOTAL"]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

import cPickle
from sklearn.decomposition import PCA
from_tfidf_pca, from_emails_key = cPickle.load(open("from_tfidf_pca.pkl", 'r'))
to_tfidf_pca, to_emails_key = cPickle.load(open("to_tfidf_pca.pkl", 'r'))

# Load tfidf-pca features.
# The features are a N-vector, and we create N separate dictionary keys in `data`,
# one for each entry in the vector.
for p in my_dataset:
    for f_idx in xrange(from_tfidf_pca.shape[1]):
        if p in from_emails_key:
            p_idx = from_emails_key.index(p)
            my_dataset[p]["from_tfidf_pca_%d" % f_idx] = from_tfidf_pca[p_idx][f_idx]
        else:
            my_dataset[p]["from_tfidf_pca_%d" % f_idx] = "NaN"
    for f_idx in xrange(to_tfidf_pca.shape[1]):
        if p in to_emails_key:
            p_idx = to_emails_key.index(p)
            my_dataset[p]["to_tfidf_pca_%d" % f_idx] = to_tfidf_pca[p_idx][f_idx]
        else:
            my_dataset[p]["to_tfidf_pca_%d" % f_idx] = "NaN"

# Add the feature into `features_list`.
for f_idx in xrange(from_tfidf_pca.shape[1]):
    features_list.append("from_tfidf_pca_%d" % f_idx)
for f_idx in xrange(to_tfidf_pca.shape[1]):
    features_list.append("to_tfidf_pca_%d" % f_idx)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Naive Bayes with Gaussian distribution.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# AdaBoost with decision tree as base estimator.
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1),
                         algorithm = "SAMME",
                         n_estimators = 40,
                         learning_rate = 2.0,
                         random_state = 13)


# Decision tree classifier with hand-picked parameters.
clf = DecisionTreeClassifier(criterion="gini",
                             max_depth = 50,
                             min_samples_split=2,
                             random_state = 73)

# Decision tree classifier, with parameters tuned with GridSearchCV.
# This is the classifier we choose for final analysis.
from sklearn.grid_search import GridSearchCV
params = {"max_depth": (10, 20, 50),
          "max_features": ("sqrt", "log2")}
clf = GridSearchCV(DecisionTreeClassifier(random_state=73), params,
                   scoring = "f1")


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.

# Find the best hyper-parameters through GridSearch.
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
clf.fit(features, labels)
clf = clf.best_estimator_

### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
