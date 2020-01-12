#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework 3.2
Gradient Boosting Classifier

@author: ariannasmith
"""

from sklearn.datasets import fetch_openml
from sklearn import ensemble as e
import time

#Use the GradientBoostingClassifier in scikit learn and try different parameters

def gradientC(xtrain, xtest, ytrain, ytest):
       
    models = (e.GradientBoostingClassifier(learning_rate = 0.1, 
            n_estimators = 100, subsample = 1.0, max_depth = 3), 
                                         
            e.GradientBoostingClassifier(learning_rate = 0.1, 
            n_estimators = 300, subsample = 1.0, max_depth = 3), 
                                         
            e.GradientBoostingClassifier(learning_rate = 0.06, 
            n_estimators = 300, subsample = 1.0, max_depth = 3),                             
    
            e.GradientBoostingClassifier(learning_rate = 0.1, 
            n_estimators = 100, subsample = 1.0, max_depth = 4), 
                                         
            e.GradientBoostingClassifier(learning_rate = 0.1, 
            n_estimators = 100, subsample = 0.6, max_depth = 3), 
                                         
            e.GradientBoostingClassifier(learning_rate = 0.1, 
            n_estimators = 100, subsample = 1.0, max_leaf_nodes = 4), 
                                         
            e.GradientBoostingClassifier(learning_rate = 0.1, 
            n_estimators = 300, subsample = 1.0, max_leaf_nodes = 4), 
                                         
            e.GradientBoostingClassifier(learning_rate = 0.06, 
            n_estimators = 300, subsample = 1.0, max_leaf_nodes = 4),                             
    
            e.GradientBoostingClassifier(learning_rate = 0.06, 
            n_estimators = 100, subsample = 1.0, max_depth = 3), 
                                         
            e.GradientBoostingClassifier(learning_rate = 0.06, 
            n_estimators = 100, subsample = 0.6, max_depth = 3))                             
    

    models = (clf.fit(xtrain, ytrain) for clf in models)
    
    print("Models fitted")
    
    scores = (clf.score(xtest, ytest) for clf in models)
    
    types = ["Learning rate- 0.1, n estimators- 100, subsample- 1.0, max depth- 3",
            "Learning rate- 0.1, n estimators- 300, subsample- 1.0, max depth- 3",
            "Learning rate- 0.06, n estimators- 300, subsample-1.0, max depth- 3",
            "Learning rate- 0.1, n estimators- 100, subsample- 1.0, max depth- 4",
            "Learning rate- 0.1, n estimators- 100, subsample- 0.6, max depth- 3",
            "Learning rate- 0.1, n estimators- 100, subsample- 1.0, max leaves- 4",
            "Learning rate- 0.1, n estimators- 300, subsample- 1.0, max leaves- 4",
            "Learning rate- 0.06, n estimators- 300, subsample-1.0, max leaves- 4",
            "Learning rate- 0.06, n estimators- 100, subsample- 1.0, max depth- 3",
            "Learning rate- 0.06, n estimators- 100, subsample- 0.6, max depth- 3"]
    
    start_time = time.time()
    
    count = 0
    for s in scores:      
        print(types[count], "score:", s)
        print("--- %s seconds ---" % round((time.time() - start_time), 2))
        count += 1
    
