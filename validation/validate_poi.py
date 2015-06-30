#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features, labels)
predictions = clf.predict(features)
print "overfit accuracy:", accuracy_score(predictions, labels)


from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size = 0.3, random_state=42)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
print "test accuracy:", accuracy_score(predictions, labels_test)
