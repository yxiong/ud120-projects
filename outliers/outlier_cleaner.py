#!/usr/bin/python

import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    error = np.abs(predictions - net_worths)
    valid = error < np.percentile(error, 90)
    cleaned_data = zip(ages[valid], net_worths[valid], error[valid])
    
    return cleaned_data

