#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 23:12:34 2019

@author: ariannasmith

Depending on your computer hardware, you may have to carefully select the 
parameters (see the documentation on scikit learn for details) in order to 
speed up the computation. Report the error rate for at least 10 parameter 
settings that you tried (see how it is reported on 
http://yann.lecun.com/exdb/mnist/). Make sure to precisely describe the 
parameters used so that your results are reproducible.

classification | pre-processing | test error rate (%)
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Use the SVM classifier in scikit learn and try different kernels and values of 
#penalty parameter.

def mySVMs(xtrain, xtest, ytrain, ytest):
    xtrain = np.ascontiguousarray(xtrain)
    xtest = np.ascontiguousarray(xtest)
    ytrain = np.ascontiguousarray(ytrain)
    ytest = np.ascontiguousarray(ytest)
    
    print("Building models...")
    models = (svm.LinearSVC(penalty = 'l1', C = 2.0, dual = False),
              svm.LinearSVC(penalty = 'l1', C = 0.5, dual = False),
              svm.LinearSVC(penalty = 'l2', C = 2.0, dual = False),
              svm.LinearSVC(penalty = 'l2', C = 0.5, dual = False),
              svm.SVC(kernel = 'rbf', C = 1.0),
              svm.SVC(kernel = 'rbf', C = 0.5),
              svm.SVC(kernel = 'poly', C = 5.0),
              svm.SVC(kernel = 'poly', C = 2.0),
              svm.SVC(kernel = 'sigmoid', C = 1.0),
              svm.SVC(kernel = 'sigmoid', C = 0.5))

    print("Fitting models...")

    models = (clf.fit(xtrain, ytrain) for clf in models)
    
    print("Scoring models...")

    scores = (clf.score(xtest, ytest) for clf in models)
    
    types = ["Linear, l1, C = 2.0", "Linear, l1, C = 0.5,", 
             "Linear, l2, C = 2.0", "Linear, l2, C = 0.5,", 
             "RBF, C = 1.0", "RBF, C = 0.5,",
             "Poly, C = 5.0", "Poly, C = 2.0,",
             "Sigmoid, C = 1.0", "Sigmoid, C = 0.5,"]
    
    count = 0
    for s in scores:      
        print(types[count], "score:", s)
        count += 1


#Use the MLPClassifier in scikit learn and try different architectures, 
#gradient descent schemes, etc




#Use the GradientBoostingClassifier in scikit learn and try different parameters






# Load data from https://www.openml.org/d/554
print("Loading data...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255
print("Data loaded.")

# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

#mySVMs(X_train, X_test, y_train, y_test)