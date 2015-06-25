#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Number of data points:", len(enron_data)
print "Number of features:", len(enron_data.itervalues().next())

poi_count = 0
for p in enron_data.values():
    if p["poi"]:
        poi_count += 1
print "Number of POIs:", poi_count

salary_count = 0
for p in enron_data.values():
    if p["salary"] != "NaN":
        salary_count += 1
print "Number of people with quantified salary:", salary_count

email_count = 0
for p in enron_data.values():
    if p["email_address"] != "NaN":
        email_count += 1
print "Number of people with email address:", email_count

no_total_payments_count = 0
for p in enron_data.values():
    if p["total_payments"] == "NaN":
        no_total_payments_count += 1
print "Number of people with no total payments:", no_total_payments_count

no_total_payments_poi_count = 0
for p in enron_data.values():
    if p["poi"] and p["total_payments"] == "NaN":
        no_total_payments_poi_count += 1
print "Number of POIs with no total payments:", no_total_payments_poi_count
