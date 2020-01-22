#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:41:33 2020

@author: Jama Hussein Mohamud
"""

# Import Needed packages 
import numpy as np
import sklearn
from Estimator import PnormRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_boston, fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


print(" this code was tested in sklearn version '0.21.2' & your sk.version is '{}'".format(sklearn.__version__))
# function to generate your data
def generate_dataset_simple(n, m, std):
  # Generate x as an array of `n` samples which can take a value between 0 and 100
  x = np.random.rand(n, m) * 100
  # Generate the random error of n samples, with a random value from a normal distribution, with a standard
  # deviation provided in the function argument
  y_intercept = np.random.randn(n) * std
  beta = np.random.rand(m)
  # Calculate `y` according to the equation discussed
  y =  np.dot(beta, x.T) + y_intercept
  return x, y

# Function to evaluate the model
def r2_score(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_tot = sum((y_true - mean_y) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# the below line is to show that the estimator has passed the sklearn estimator checker. 

print("-------The estimator has not passed all the checks-----" if check_estimator(PnormRegressor) else 
      "-------The estimator has passed all the checks-------")

# generate X, y randomly from the above function 
X, y = generate_dataset_simple(100, 4, 0.25)
# Normalizing the data the data
X = (X - X.mean(axis=0))/X.std(axis=0)

#num_iterations, learning_rate , p_norm = 100, 1e-5, 2

regressor  = PnormRegressor()
regressor.fit(X, y)
pred = regressor.predict(X) 

print('Train R2_score for randomly generated data: ', r2_score(y, pred))

#######################################################################################
print('---- Using Datasets availlable in sklearn like BOSTON dataset for housing price prediction---')

X, y = load_boston(return_X_y=True)

# Please normalize the data
X = (X - X.mean(axis=0))/X.std(axis=0)

regressor.fit(X, y)  

pred = regressor.predict(X)
print('Train set R2_score for BOSTON data: ', r2_score(y, pred))

##################################################################################
print('----using sklearn packages like GridSearch to search for best parameters---')

tuned_params = {"num_iterations": [100,1000], "p_norm" : [1,2]}

pipe = GridSearchCV(PnormRegressor(), tuned_params)
pipe.fit(X,y)

pred = pipe.predict(X) 
print('Train set R2_score for BOSTON data with gridsearch: ', r2_score(y, pred))

print("Best parameters chose by GridSearch: ", pipe.best_params_)


#####################################################################################
print('---Using sklearn-openML interface to evaluate the model on other datasets.---')

X,y = fetch_openml(name='boston', return_X_y=True)

# Normalizing the data
X = (X - X.mean(axis=0))/X.std(axis=0)

regressor.fit(X, y)  

pred = regressor.predict(X)
print('Train set R2_score for BOSTON data fetched from openML: ', r2_score(y, pred))

#######################################################################################
print('----- MODEL EVALUATION on  Employee Selection (ESL) dataset fetched from openML DB')

# Fetch Employee Selection dataset from openML
X,y = fetch_openml(name='ESL', return_X_y=True)

# Normalizing the data
X = (X - X.mean(axis=0))/X.std(axis=0)

# split data into train and validation
train_split_perc = 0.8
trainset = round(train_split_perc * len(X))
X_train, X_test, y_train, y_test = X[:trainset], X[trainset:], y[:trainset], y[trainset:]

model = PnormRegressor()

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("R2_score for ESL train data: {}".format(r2_score(y_train, y_pred_train)))
print("R2_score for ESL test data: {}".format(r2_score(y_test, y_pred_test)))

