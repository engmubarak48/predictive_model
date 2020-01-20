#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:23:08 2020

@author: mubarak
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#%%
# generating data 
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

#%%

def hypothesis(X, beta):
    return np.dot(X, beta)

def loss_function(X, y, beta, p_norm = 2):
    m = len(y)
    y_predicted = hypothesis(X, beta)
    return np.sum(1/(2 * m) * np.abs(y - y_predicted) ** p_norm)
#
#def gradients(X, y, beta, p_norm = 2):
#    y_predicted = hypothesis(X, beta)
#    loss = y - y_predicted
#    L_beta = p_norm/len(y) * np.dot(loss/((np.abs(loss)) ** (p_norm - 1)), -X)
#    return L_beta

def gradients(X, y, beta, p_norm = 2):
    y_predicted = hypothesis(X, beta)
    loss = y - y_predicted
    L_beta = p_norm * np.dot(((np.abs(loss)**p_norm)/loss), -X)
    return L_beta

def gradient_descent(X, y, num_iterations, learning_rate, p_norm = 2):
    np.random.seed(1)
    losses = np.zeros(num_iterations)
    X = (X - X.mean(axis=0))/X.std(axis=0)
    X = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.random.rand(X.shape[1])
    for i in range(num_iterations):
        L_beta = gradients(X, y, beta, p_norm = 2)
        beta = beta - learning_rate * L_beta
        cost = loss_function(X, y, beta, p_norm = 2)
        losses[i] = cost
    return losses, beta

def predict(X, beta):
    X = np.column_stack([np.ones(X.shape[0]), X])
    return np.dot(X, beta)

#%%
#X, y = generate_dataset_simple(100, 4, 0.25)
##beta  = np.random.rand(X.shape[1])
num_iterations, learning_rate = 1000, 1e-5


from sklearn.datasets import load_boston
#from sklearn.pipeline import make_pipeline

X, y = load_boston(return_X_y=True)


losses, beta = gradient_descent(X, y, num_iterations, learning_rate, p_norm = 2)





