'''
This script tests and validates (with five fold cross validation) various machine learning models on the .npz files of numpy
binary data. The data used is a set of gray scale images (32 × 32 pixels) with one (and only one) of the following objects: 
horse, truck, frog, ship (labels 0, 1, 2 and 3, respectively) and contains 20,000 rows for sampling and 3,346 rows for testing.
'''

#Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score
import sklearn.tree as tree
import sklearn.model_selection as ms
import sklearn.neighbors as nb
import sklearn.linear_model as log
import autograd.numpy as npa
import autograd
from autograd.util import flatten
import matplotlib.pyplot as plt
import time

#Function to read in train and test image data
def read_image_data():
	print('Reading image data ...')
	temp = np.load('data_train.npz')
	train_x = temp['data_train']
	temp = np.load('labels_train.npz')
	train_y = temp['labels_train']
	temp = np.load('data_test.npz')
	test_x = temp['data_test']
	return (train_x, train_y, test_x)

#Get train and test image data
train_x, train_y, test_x = read_image_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

######################################### DECISION TREES WITH FIVE-FOLD CROSS VALIDATION #########################################
#Specify k-fold criteria
kf = ms.KFold(n_splits = 5, random_state = 69, shuffle = True)

#Instantiate tree depths to test
testDepths = [3, 6, 9, 12, 14]

#Cross validate 5 different decision trees for image id dataset using a for loop
for i in testDepths:
    #Loop through training decision trees with different depths
    for train_index, test_index in kf.split(train_x):
        clf = tree.DecisionTreeClassifier(max_depth = i, random_state = 27)
        clf = clf.fit(train_x[train_index], train_y[train_index])
        pred = clf.predict(train_x[test_index])
        
        #Take error by subtracting accuracy from 1
        error = 1 - accuracy_score(pred, train_y[test_index], normalize = True) 
        
    #Print results of cross validation
    print('max depth: %s, estimated error: %s' %(i, error))

############################# TRAIN SELECTED TREE
#Learn the tree with max depth 9
clfBest = tree.DecisionTreeClassifier(max_depth = 9, random_state = 27)
clfBest.fit(train_x, train_y)
predicted_y = clfBest.predict(test_x)

#Compute error for chosen model
pred = clfBest.predict(train_x)
bestError = accuracy_score(pred, train_y, normalize = True)
print ('The best estimated test accuracy: %s' %(bestError))

############################################ KNN WITH FIVE-FOLD CROSS VALIDATION #############################################
#Specify k-fold criteria
kf = ms.KFold(n_splits = 5, random_state = 69, shuffle = True)

#Instantiate tree depths to test
testNeighbors = [3, 5, 7, 9, 11]

#Cross validate 5 different decision trees for image id dataset using a for loop
for i in testNeighbors:
    #Loop through training decision trees with different depths
    for train_index, test_index in kf.split(train_x):
        clf = nb.KNeighborsClassifier(n_neighbors = i)
        clf = clf.fit(train_x[train_index], train_y[train_index])
        pred = clf.predict(train_x[test_index])
        
        #Take error by subtracting accuracy from 1
        error = 1 - accuracy_score(pred, train_y[test_index], normalize = True) 
        
    #Print results of cross validation
    print('# of neighbors: %s, estimated error: %s' %(i, error))

############################# TRAIN SELECTED KNN
#Learn KNN with k = 3
clfBest = nb.KNeighborsClassifier(n_neighbors = 3)
clfBest.fit(train_x, train_y)
predicted_y = clfBest.predict(test_x)

#Compute error for chosen model
pred = clfBest.predict(train_x)
bestError = accuracy_score(pred, train_y, normalize = True)
print ('The best estimated test accuracy: %s' %(bestError))

############################ LINEAR MODEL WITH L2 REGULARLIZATION WITH FIVE-FOLD CROSS VALIDATION ############################
#Specify k-fold criteria
kf = ms.KFold(n_splits = 5, random_state = 69, shuffle = True)

#Instantiate test values
testLoss = ['log', 'hinge']
testAlpha = [0.00001, 0.001, 0.1, 1, 10]

#Cross validate 10 different logistic models for image id dataset using a for loop
for j in testLoss:
    for i in testAlpha:
        #Loop through training decision trees with different depths
        for train_index, test_index in kf.split(train_x):
            clf = log.SGDClassifier(loss = j, penalty = 'l2', alpha = i)
            clf = clf.fit(train_x[train_index], train_y[train_index])
            pred = clf.predict(train_x[test_index])
        
            #Take error by subtracting accuracy from 1
            error = 1 - accuracy_score(pred, train_y[test_index], normalize = True) 
        
        #Print results of cross validation
        print('Loss: %s, alpha: %s, estimated error: %s' %(j, i, error))

############################# TRAIN SELECTED LOGISIC REGRESSION
#Learn logistic regression with logistic loss and alpha = 0.1
clfBest = log.SGDClassifier(loss = 'log', penalty = 'l2', alpha = 0.1)
clfBest.fit(train_x, train_y)
predicted_y = clfBest.predict(test_x)

#Compute error for chosen model
pred = clfBest.predict(train_x)
bestError = accuracy_score(pred, train_y, normalize = True)
print ('The best estimated test accuracy: %s' %(bestError))

################################################## NEURAL NET TRAINING ##################################################
#Load training labels
train_y_integers = temp['labels_train']
yTrainInt = temp['labels_train']

############################# NEURAL NET TRAINING
#Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
train_x -= .5
test_x  -= .5

#Number of output dimensions
dims_out = 4
#Number of hidden units
dims_hid = [5, 40, 70]
#Learning rate
epsilon = 0.0001
#Momentum of gradients update
momentum = 0.1
#Number of epochs
nEpochs = range(0, 1001, 1)
#Number of train examples
nTrainSamples = train_x.shape[0]
#Number of input dimensions
dims_in = train_x.shape[1]

#Convert integer labels to one-hot vectors
#i.e. convert label 2 to 0, 0, 1, 0
train_y = npa.zeros((nTrainSamples, dims_out))
train_y[npa.arange(nTrainSamples), train_y_integers] = 1

assert momentum <= 1
assert epsilon <= 1

#Empty list to store mean logistic loss and runtime
meanLoss = []
runTimes = list()

for i in dims_hid:
    #Start timer
    start = time.time()
    
    #Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
    grad_fun = autograd.grad_and_aux(logistic_loss_batch)
    
    #Initialize weights
    W = npa.random.randn(dims_in, i)
    b = npa.random.randn(i)
    V = npa.random.randn(i, dims_out)
    c = npa.random.randn(dims_out)
    smooth_grad = 0
    
    #Compress all weights into one weight vector using autograd's flatten
    all_weights = (W, b, V, c)
    weights, unflatten = flatten(all_weights)

    for j in nEpochs:
        #Compute gradients (partial derivatives) using autograd toolbox
        weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
        meanLoss.append(returned_values[0]/nTrainSamples)
        
        #Update weight vector
        smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
        weights = weights - epsilon * smooth_grad
    
    #Get most updated weight vector
    weightsFinal = weights
    
    #Add time to runTime
    end = time.time() #End timer
    timeTaken = (end - start)
    runTime.append(timeTaken)
        
    #Print diagnostics for each NN
    print ('Hidden units:', i, '\n',
           'Training time:', runTime[-1], '\n',
           'Logistic loss: ', meanLoss[-1], '\n',
           'Train accuracy: ', 1-mean_zero_one_loss(weightsFinal, train_x, train_y_integers, unflatten))

############################# PLOT TRAINING TIMES
#Create values and labels for line graphs
values = meanLoss
epochs = range(0, 1000, 1)
labels = ["Hidden layer size 5", "Hidden layer size 40", "Hidden layer size 70"]

#Plot a line graph
plt.figure(2, figsize=(7, 5))  #6x4 is the aspect ratio for the plot
plt.plot(epochs, values[0:1000], color = "#4B0082", linewidth = 2) 
plt.plot(epochs, values[1000:2000], color = "#228B22", linewidth = 2)
plt.plot(epochs, values[2000:3000], color = "#20B2AA", linewidth = 2)

#This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Mean logistic loss") #Y-axis label
plt.xlabel("Epoch") #X-axis label
plt.title("Mean Logistic Loss for Neural Network Training") #Plot title
plt.xlim(0, 1000) #set x axis range
plt.ylim(0, 50) #Set yaxis range
plt.legend(labels, loc = "best")

#Displays the plots
#You must close the plot window for the code following each show()
#to continue to run
plt.show()


############################# PICK BEST NETWORK
#Split into stratified groups
stSplit = ms.StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, train_size = 0.8, random_state = 69)

for train_index, test_index in stSplit.split(train_x, train_y):
    xTrain, xTest = train_x[train_index], train_x[test_index]
    yTrain, yTest = train_y[train_index], train_y[test_index]

#Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
xTrain -= .5
xTest  -= .5

#Number of output dimensions
dims_out = 4
#Number of hidden units
dims_hid = [5, 40, 70]
#Learning rate
epsilon = 0.0001
#Momentum of gradients update
momentum = 0.1
#Number of epochs
nEpochs = range(0, 1001, 1)
#Number of train examples
nTrainSamples = xTrain.shape[0]
#Number of input dimensions
dims_in = xTrain.shape[1]

#Convert integer labels to one-hot vectors
#i.e. convert label 2 to 0, 0, 1, 0
yTrain = npa.zeros((nTrainSamples, dims_out))
yTrain[npa.arange(nTrainSamples), train_y_integers[train_index]] = 1

assert momentum <= 1
assert epsilon <= 1

#Empty list to store mean logistic loss
meanLoss = []

for i in dims_hid:
    #Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
    grad_fun = autograd.grad_and_aux(logistic_loss_batch)
    
    #Initialize weights
    W = npa.random.randn(dims_in, i)
    b = npa.random.randn(i)
    V = npa.random.randn(i, dims_out)
    c = npa.random.randn(dims_out)
    smooth_grad = 0
    
    #Compress all weights into one weight vector using autograd's flatten
    all_weights = (W, b, V, c)
    weights, unflatten = flatten(all_weights)

    for j in nEpochs:
        #Compute gradients (partial derivatives) using autograd toolbox
        weight_gradients, returned_values = grad_fun(weights, xTrain, yTrain, unflatten)
        meanLoss.append(returned_values[0]/nTrainSamples)
        
        #Update weight vector
        smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
        weights = weights - epsilon * smooth_grad
    
    #Get most updated weight vector
    weightsFinal = weights
        
    #Print diagnostics for each NN
    print ('Hidden units:', i, '\n',
           'Estimated test error:', mean_zero_one_loss(weightsFinal, xTrain, train_y_integers[train_index], unflatten))

############################# TRAIN BEST NEURAL NET
#Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
train_x -= .5
test_x  -= .5

#Number of output dimensions
dims_out = 4
#Number of hidden units
dims_hid = 70
#Learning rate
epsilon = 0.0001
#Momentum of gradients update
momentum = 0.1
#Number of epochs
nEpochs = range(0, 1001, 1)
#Number of train examples
nTrainSamples = train_x.shape[0]
#Number of input dimensions
dims_in = train_x.shape[1]

#Convert integer labels to one-hot vectors
#i.e. convert label 2 to 0, 0, 1, 0
train_y = npa.zeros((nTrainSamples, dims_out))
train_y[npa.arange(nTrainSamples), train_y_integers] = 1

assert momentum <= 1
assert epsilon <= 1


#Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
grad_fun = autograd.grad_and_aux(logistic_loss_batch)

#Initialize weights
W = npa.random.randn(dims_in, dims_hid)
b = npa.random.randn(dims_hid)
V = npa.random.randn(dims_hid, dims_out)
c = npa.random.randn(dims_out)
smooth_grad = 0

#Compress all weights into one weight vector using autograd's flatten
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)

for j in nEpochs:
    #Compute gradients (partial derivatives) using autograd toolbox
    weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)

    #Update weight vector
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad

#Get most updated weight vector
weightsFinal = weights

#Make predictions
(W, b, V, c) = unflatten(weightsFinal)
out = feedForward(W, b, V, c, test_x)
pred = npa.argmax(out, axis = 1)