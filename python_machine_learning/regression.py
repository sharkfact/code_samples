'''
This script uses my handwritten cross validation grid search function to test and validate (with five fold cross validation)
various machine learning models on the .txt files of data. The data used is from a power plant and contains 7176 rows for sampling
and 2,392 rows for testing. Its variables are temperature, pressure, humidity, exhaust vacuum, and electrical energy output as
the dependent variable. 
'''

#Import python modules
import numpy as np
import copy as cp
import sklearn.tree as tree
import sklearn.model_selection as ms
import sklearn.linear_model as lin
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn.neighbors as nb
import time
from pylab import *

#Function to read in train and test power plant data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	ppTrain_x = np.loadtxt('data_train.txt')
	ppTrain_y = np.loadtxt('labels_train.txt')
	ppTest_x = np.loadtxt('data_test.txt')

	return (ppTrain_x, ppTrain_y, ppTest_x)

#Function to compute MAE
def compute_error(y_hat, y):
	#Mean absolute error
	return np.abs(y_hat - y).mean()

#Get train and test power plant data
ppTrain_x, ppTrain_y, ppTest_x = read_data_power_plant()
print('Train =', ppTrain_x.shape)
print('Test =', ppTest_x.shape)

#Create dummy test output values
ppPredicted_y = np.ones(ppTest_x.shape[0]) * -1

######################################### DECISION TREES WITH FIVE-FOLD CROSS VALIDATION #########################################
#Specify k-fold criteria
kf = ms.KFold(n_splits = 5, random_state = 69, shuffle = True)

#Instantiate values to call later
testNeighbors = [3, 5, 10, 20, 25]

#Cross validate 5 different decision trees for power plant dataset using a for loop
for i in testNeighbors: 
    #Loop through training KNN with different values for K
    for train_index, test_index in kf.split(ppTrain_x):
        clf = nb.KNeighborsRegressor(n_neighbors = i)
        clf = clf.fit(ppTrain_x[train_index], ppTrain_y[train_index])
        pred = clf.predict(ppTrain_x[test_index])
        
        #Calculate error
        error = compute_error(pred, ppTrain_y[test_index]) #Take mean absolute error
    
    #Print results of cross validation
    print('# of neighbors: %s, estimated sample error: %s' %(i, error))

############################# TRAIN SELECTED MODEL
#Learn KNN with K = 3
clfBest = nb.KNeighborsRegressor(n_neighbors = 3)
clfBest.fit(ppTrain_x, ppTrain_y)
ppPredicted_y = clfBest.predict(ppTest_x)

#Compute MAE for chosen model
ppPred = clfBest.predict(ppTrain_x)
bestError = compute_error(ppPred, ppTrain_y)
print ('The lowest estimated test error is: %s' %(bestError))

######################################## RIDGE REGRESSION WITH FIVE-FOLD CROSS VALIDATION ########################################
#Specify k-fold criteria
kf = ms.KFold(n_splits = 5, random_state = 69, shuffle = True)

#Instantiate values to call later
testAlphas = [0.00001, 0.001, 0.1, 1, 10]

#Cross validate ridge regression for power plant dataset using a for loop
for i in testAlphas:
    #Loop ridge regularization regression with different values of alpha
    for train_index, test_index in kf.split(ppTrain_x):
        clf = lin.Ridge(alpha = i)
        clf = clf.fit(ppTrain_x[train_index], ppTrain_y[train_index])
        pred = clf.predict(ppTrain_x[test_index])
        
        #Calculate error
        error = compute_error(pred, ppTrain_y[test_index]) #Take mean absolute error
        
    #Print results of cross validation
    print('Ridge alpha: %s, estimated sample error: %s' %(i, error))

#Cross validate lasso regression for power plant dataset using a for loop
for i in testAlphas:
    #Loop Lasso regularization regression with different values of alpha
    for train_index, test_index in kf.split(ppTrain_x):
        clf = lin.Lasso(alpha = i)
        clf = clf.fit(ppTrain_x[train_index], ppTrain_y[train_index])
        pred = clf.predict(ppTrain_x[test_index])
        
        #Calculate error
        error = compute_error(pred, ppTrain_y[test_index]) #Take mean absolute error
        
    #Print results of cross validation
    print('Lasso alpha: %s, estimated sample error: %s' %(i, error))

############################# TRAIN SELECTED MODEL
#Learn KNN with alpha = 0.00001
clfBest = lin.Ridge(alpha = 0.00001)
clfBest.fit(ppTrain_x, ppTrain_y)
ppPredicted_y = clfBest.predict(ppTest_x)

#Compute MAE for chosen model
ppPred = clfBest.predict(ppTrain_x)
bestError = compute_error(ppPred, ppTrain_y)
print ('The lowest estimated test error is: %s' %(bestError))       

######################################## LASSO REGRESSION WITH FIVE-FOLD CROSS VALIDATION ########################################  
#Cross validate lasso regression for power plant dataset using a for loop
for i in testAlphas:
    #Loop Lasso regularization regression with different values of alpha
    for train_index, test_index in kf.split(locTrain_x):
        clf = lin.Lasso(alpha = i)
        clf = clf.fit(ppTrain_x[train_index], ppTrain_y[train_index])
        pred = clf.predict(pprain_x[test_index])
        
        #Calculate error
        error = compute_error(pred, ppTrain_y[test_index]) #Take mean absolute error
        
    #Print results of cross validation
    print('Lasso alpha: %s, estimated sample error: %s' %(i, error))

############################# TRAIN SELECTED MODEL
#Learn lasso regression with alpha = 0.001
clfBest = lin.Lasso(alpha = 0.001)
clfBest.fit(ppTrain_x, ppTrain_y)
ppPredicted_y = clfBest.predict(ppTest_x)

#Compute MAE for chosen model
locPred = clfBest.predict(ppTrain_x)
bestError = compute_error(ppPred, ppTrain_y)
print ('The lowest estimated test error is: %s' %(bestError))


########################################################### GRID SEARCH ##########################################################
#PREPARATION
#Decision tree regression function for given hyperparameter value
def clfTree(hyp):
    clf = tree.DecisionTreeRegressor(max_depth = hyp, criterion = 'mae')
    return clf

#KNN regression function for given hyperparameter value
def clfKnn(hyp):
    clf = nb.KNeighborsRegressor(n_neighbors = hyp)
    return clf

#Logistic regression function with lasso regularization given hyperparameter value
def clfLasso(hyp):
    clf = lin.Lasso(alpha = hyp)
    return clf

#Logistic regression function with lasso regularization given hyperparameter value
def clfRidge(hyp):
    clf = lin.Ridge(alpha = hyp)
    return clf

#Dictionary of classifier names (keys) and their classification functions (vals) from above
clfDict = {'tree': clfTree, 'KNN': clfKnn, 'lasso': clfLasso, 'ridge': clfRidge}

'''
Grid search given a classifier, parameters and training data
Of a set or range of hyperparameter values, return the value with the lowest MAE for a given classifier
clfChoice: a string corresponding to the chosen classifier('Decision Tree', 'KNN', 'Lasso Regression', 'Ridge Regression')
trainFeats: the training data
hyperParams: a tuple of values to test for a hyperparameter
nFolds: the total number of folds for partitioning the training data
hypRange: boolean True or False.
    If True and hyperParams contains three values, they are treated like range(start, stop, step)
    If False, each value in hyperParams is evaluated individually
'''
def gridSearch(clfChoice, hyperParams, hypRange, trainFeats, trainLabs, nFolds):
    #Set cross validation parameters specified by user
    kf = ms.KFold(n_splits = nFolds, random_state = 69)
    
    results = {} #Initialize dictionary to hold results
    
    #Check to see how to identify hyperparameter values to test
    if hypRange:
        hyperParams = np.arange(*hyperParams) #Use numpy's arrange() function to enable non-integer ranges

    #Loop through hyperparameter values to test
    for hyp in hyperParams:
        
        #Cross validate and collect MAE
        for train_index, test_index in kf.split(trainFeats):
            
            #Get classifier specified by the user for this hyperparameter value
            clf = clfDict[clfChoice](hyp)
            clf = clf.fit(trainFeats[train_index], trainLabs[train_index])
            predict = clf.predict(trainFeats[train_index])
            
            #Compute MAE
            errors = compute_error(predict, trainLabs[train_index])
            
        #Print error for each run to track function progress
        print(errors)
        
        #Make new dictionary for each hyperparameter value and its MAE
        hypErr = {hyp:errors}
        results.update(hypErr) #Append dictionary for each loop to results dictionary
    
    #Select best hyperparameter value and associated MAE
    bestHyp = min(results, key = results.get)
    bestMAE = results[bestHyp]
     
    return print('The best hyperparameter value for %s regression is %s with an estimated sample MAE of %s' %(clfChoice, bestHyp, bestMAE))


############################# SELECT BEST MODEL FOR POWER PLANT DATA
gridSearch(clfChoice = 'tree',
           hyperParams = (5, 11, 1),
           hypRange = True,
           trainFeats = ppTrain_x,
           trainLabs = ppTrain_y,
           nFolds = 5)
           
############################# TRAIN SELECTED MODEL
#Specify k-fold criteria
kf = ms.KFold(n_splits = 5, random_state = 69, shuffle = True)

clfBest = tree.DecisionTreeRegressor(max_depth = 10, criterion = 'mae')
clfBest.fit(ppTrain_x, ppTrain_y)
ppPredicted_y = clfBest.predict(ppTest_x)

#Compute MAE for chosen model
ppPred = clfBest.predict(ppTrain_x)
bestError = compute_error(ppPred, ppTrain_y)
print ('The lowest estimated test error is: %s' %(bestError))
