#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework 3.2
MLP Classifier

@author: ariannasmith
"""

from sklearn.neural_network import MLPClassifier
import warnings
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


#Use the MLPClassifier in scikit learn and try different architectures, 
#gradient descent schemes, etc

def neuralC(xtrain, xtest, ytrain, ytest):
  
    print("Building models...")
    models = (MLPClassifier(hidden_layer_sizes = (50,), activation = 'logistic', 
              solver ='sgd', max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (200,), activation = 'logistic', 
              solver ='sgd', max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (50,), activation = 'tanh', 
              solver ='sgd', max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (200,), activation = 'tanh', 
              solver ='sgd', max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (50,), activation = 'logistic', 
              solver ='adam', max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (200,), activation = 'logistic', 
              solver ='adam',max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (50,), activation = 'tanh', 
              solver ='adam',max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (200,), activation = 'tanh', 
              solver ='adam',max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (50,), activation = 'relu', 
              solver ='adam',max_iter = 400),
            MLPClassifier(hidden_layer_sizes = (200,), activation = 'relu', 
              solver ='adam',max_iter = 400))

    models = (clf.fit(xtrain, ytrain) for clf in models)
    
    print("Scoring models...")

    scores = (clf.score(xtest, ytest) for clf in models)
    
    types = ["50 hidden layers, Logistic, SGD,", 
             "200 hidden layers, Logistic, SGD,",
             "50 hidden layers, tanh, SGD,", 
             "200 hidden layers, tanh, SGD,",
             "50 hidden layers, Logistic, adam,", 
             "200 hidden layers, Logistic, adam,",
             "50 hidden layers, tanh, adam,", 
             "200 hidden layers, tanh, adam,",
             "50 hidden layers, relu, adam,", 
             "200 hidden layers, relu, adam,",]


    count = 0
    for s in scores:      
        print(types[count], "score:", s)
        count += 1
        
