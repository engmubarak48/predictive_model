# predictive_model

This repository is an implementation of a regression predictive model. The model is sklearn compatible and it has passed sklearn estimator check.
It can be used together with all other functionalities in sklearn like GridSearch, get_params, set_params and others. 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:41:33 2020

@author: Jama Hussein Mohamud
"""

# How to USE
import numpy as np
from Estimator import PnormRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_boston
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
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


# the below line is to show that the estimator has passed the sklearn estimator checker. 

print("the estimator has passed all the checks" if check_estimator(PnormRegressor) else 
      "the estimator has not passed all the checks")

# generate X, y randomly from the above function 
X, y = generate_dataset_simple(100, 4, 0.25)
# Please normalize the data
X = (X - X.mean(axis=0))/X.std(axis=0)

num_iterations, learning_rate , p_norm = 100, 1e-5, 2
regressor  = PnormRegressor()
estimator = regressor.fit(X, y)
pred = regressor.predict(X) 

print('Train R2_score for randomly generated data', r2_score(y, pred))

# We can also use Datasets availlable in sklearn like BOSTON dataset for housing price prediction

X, y = load_boston(return_X_y=True)

# Please normalize the data
X = (X - X.mean(axis=0))/X.std(axis=0)

regressor.fit(X, y)  

pred = regressor.predict(X)
print('Train set R2_score for BOSTON data', r2_score(y, pred))

# We can also use other sklearn packages like GridSearch to search best parameters

tuned_params = {"num_iterations": [100,1000], "p_norm" : [1,2]}

pipe = GridSearchCV(PnormRegressor(), tuned_params)
pipe.fit(X,y)

pred = pipe.predict(X) 
print('Train set R2_score for BOSTON data with gridsearch', r2_score(y, pred))

print("Best parameters chose by GridSearch", pipe.best_params_)


