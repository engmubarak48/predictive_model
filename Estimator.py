#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:21:09 2020

@author: Jama Hussein Mohamud
"""
import numpy as np
np.seterr(all='ignore')
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

        def loss_function(X, y, beta):
            m = len(y)
            y_predicted = hypothesis(X, beta)
            return np.sum(1/(2 * m) * np.abs(y - y_predicted) ** self.p_norm)
        
        def gradients(X, y, beta):
            y_predicted = hypothesis(X, beta)
            loss = y - y_predicted
            L_beta = self.p_norm * np.dot(((np.abs(loss)** self.p_norm)/loss), -X)
            return L_beta
    
        def gradient_descent(X, y):
            np.random.seed(self.random_state)
            losses_ = np.zeros(self.num_iterations)
            X = np.column_stack([np.ones(X.shape[0]), X])
            beta_ = np.random.rand(X.shape[1])
            for i in range(self.num_iterations):
                L_beta = gradients(X, y, beta_)
                beta_ = beta_ - self.learning_rate * L_beta
                cost = loss_function(X, y, beta_)
                losses_[i] = cost
            return losses_, beta_
        self.losses_, self.beta_ = gradient_descent(X, y)
        self.X_ = X
        self.y_ = y
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # Input validation
        X = check_array(X)
        X = np.column_stack([np.ones(X.shape[0]), X])
        predictions = np.dot(X, self.beta_)
        return predictions




