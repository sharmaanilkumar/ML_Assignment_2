#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:05:35 2019

@author: anil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('winequality-red.csv', sep=";")
df1 = pd.read_csv('winequality-white.csv', sep = ";")
len(df1)
X1 = df1.drop(['quality'], axis = 1)
Y1=df1['quality'].values.reshape((4898,1))

X=df.drop(['quality'],axis=1)
Y=df['quality'].values.reshape((1599,1))
theta=np.zeros([1,12])
theta = theta.reshape((12,1))


one = np.ones((1599,1))
X = np.concatenate((one,X),axis=1)
    
one1 = np.ones((4898,1))
X1 = np.concatenate((one1,X1),axis=1)

def  cal_cost(theta,X,Y):
    m = len(Y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions-Y))
    return cost

def gradient_descent(X,Y,theta,learning_rate,iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,12))
    for it in range(iterations):
        
        predictions = np.dot(X,theta)
        
        theta = theta -(1/m)*learning_rate*( X.T.dot((predictions - Y)))
        theta_history[it,:] =theta.T
        cost_history[it]  = cal_cost(theta,X,Y)
        
    return theta, cost_history, theta_history

learning_rate = 0.000001
iterations = 100000

theta1,c,th = gradient_descent(X,Y,theta,learning_rate,iterations)

n = [i for i in range(iterations)]
plt.plot(n,c)

pred = np.dot(X1, theta1)


for j in range(len(pred)):
    if pred[j] < 1.5:
        pred[j] = 1
    elif (1.5 <= pred[j] < 2.5):
        pred[j] = 2
    elif (2.5 <= pred[j] < 3.5):
        pred[j] = 3
    elif (3.5 <= pred[j] < 4.5):
        pred[j] = 4
    elif (4.5 <= pred[j] < 5.5):
        pred[j] = 5
    elif (6.5 <= pred[j] < 7.5):
        pred[j] = 7
    else:
        pred[j] = 8
    
for j in range(len(pred)):
    print(pred[j])
