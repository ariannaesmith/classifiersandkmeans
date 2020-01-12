#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework 3.2

@author: ariannasmith
"""

from sklearn.datasets import fetch_openml
import sys
import svms
import neurals
import gradient


# Load data from https://www.openml.org/d/554
print("Loading data...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255
print("Data loaded.")

# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


if sys.argv[1] == "SVM":
    print("Running SVM Classifiers")
    svms.SVMC(X_train, X_test, y_train, y_test)
elif sys.argv[1] == "MLP":
    print("Running MLP Classifier")
    neurals.neuralC(X_train, X_test, y_train, y_test)
elif sys.argv[1] == "GB":
    print("Running Gradient Boosting Classifier")
    gradient.gradientC(X_train, X_test, y_train, y_test)
else:
    print("Incorrect command- please try again.")


    