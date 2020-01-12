#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework 3.2
SVM Classifiers

@author: ariannasmith

"""

import numpy as np
from sklearn import svm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Use the SVM classifier in scikit learn and try different kernels and values of 
#penalty parameter.

def SVMC(xtrain, xtest, ytrain, ytest):
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

