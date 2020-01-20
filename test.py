#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:08:41 2020

@author: mubarak
"""
# import libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

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

def gradients(X, y, beta, p_norm = 2):
    y_predicted = hypothesis(X, beta)
    L_beta = np.dot((p_norm/(len(y)) * (y - y_predicted) ** (p_norm -1)), -X)
    return L_beta

def gradient_descent(X, y, num_iterations, learning_rate, p_norm = 2):
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
    X = (X - X.mean(axis=0))/X.std(axis=0)
    X = np.column_stack([np.ones(X.shape[0]), X])
    return np.dot(X, beta)

#%%
X, y = generate_dataset_simple(100, 4, 0.25)
#beta  = np.random.rand(X.shape[1])
num_iterations, learning_rate = 100, 1e-5

#%%
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

def gradient_descents(X, Y, alpha, iterations):
    B = np.zeros(X.shape[1])
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history







































