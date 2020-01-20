#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:21:09 2020

@author: mubarak
"""
import numpy as np
np.seterr(all='ignore')
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

class PnormRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, num_iterations=1000, learning_rate=1e-5, p_norm = 2, random_state=1):
        self.random_state = random_state
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.p_norm = p_norm    
        
    def fit(self, X, y):
        X,y = check_X_y(X, y)
        
        def hypothesis(X, beta):
            return np.dot(X, beta)

        def loss_function(X, y, beta, p_norm):
            m = len(y)
            y_predicted = hypothesis(X, beta)
            return np.sum(1/(2 * m) * np.abs(y - y_predicted) ** p_norm)
        
        def gradients(X, y, beta, p_norm = 2):
            y_predicted = hypothesis(X, beta)
            loss = y - y_predicted
            L_beta = p_norm * np.dot(((np.abs(loss)**p_norm)/loss), -X)
            return L_beta
    
        def gradient_descent(X, y, num_iterations, learning_rate, p_norm):
            np.random.seed(self.random_state)
            losses_ = np.zeros(num_iterations)
            X = np.column_stack([np.ones(X.shape[0]), X])
            beta_ = np.random.rand(X.shape[1])
            for i in range(self.num_iterations):
                L_beta = gradients(X, y, beta_, p_norm)
                beta_ = beta_ - learning_rate * L_beta
                cost = loss_function(X, y, beta_, p_norm)
                losses_[i] = cost
            return losses_, beta_
        self.losses_, self.beta_ = gradient_descent(X, y, self.num_iterations, self.learning_rate, self.p_norm)
        self.X_ = X
        self.y_ = y
        return self
#    def get_params(self, deep=True):
#        # suppose this estimator has parameters "alpha" and "recursive"  
#        return {"num_iterations": self.num_iterations, "learning_rate": self.learning_rate, "p_norm": self.p_norm}
#
#    def set_params(self, **parameters):
#        for parameter, value in parameters.items():
#            setattr(self, parameter, value)
#        return self
    def score(self, X, Y):
        
        Y_pred = self.predict(X)
        mean_y = np.mean(Y)
        ss_tot = sum((Y - mean_y) ** 2)
        ss_res = sum((Y - Y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
#        print(r2)
        return r2
#    def _more_tags(self):
#        return {'poor_score': True}
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # Input validation
        X = check_array(X)
#        X = (X - X.mean(axis=0))/X.std(axis=0)
        X = np.column_stack([np.ones(X.shape[0]), X])
        predictions = np.dot(X, self.beta_)
        return predictions

check_estimator(PnormRegressor)
#%%
#def generate_dataset_simple(n, m, std):
#  # Generate x as an array of `n` samples which can take a value between 0 and 100
#  x = np.random.rand(n, m) * 100
#  # Generate the random error of n samples, with a random value from a normal distribution, with a standard
#  # deviation provided in the function argument
#  y_intercept = np.random.randn(n) * std
#  beta = np.random.rand(m)
#  # Calculate `y` according to the equation discussed
#  y =  np.dot(beta, x.T) + y_intercept
#  return x, y
#
#X, y = generate_dataset_simple(100, 4, 0.25)
##beta  = np.random.rand(X.shape[1])
#num_iterations, learning_rate = 100, 1e-5
#regressor  = PnormRegressor(num_iterations=num_iterations, learning_rate=learning_rate)
#
#estimator = regressor.fit(X, y)

from sklearn.datasets import load_boston
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

X, y = load_boston(return_X_y=True)
tuned_params = {"p_norm" : [1,2,4]}

pipe = GridSearchCV(PnormRegressor(), tuned_params)


#pipe = make_pipeline(PnormRegressor())
pipe.fit(X, y)  

#
#pred = pipe.predict(X)  
#
#pipe.score(X, y)
