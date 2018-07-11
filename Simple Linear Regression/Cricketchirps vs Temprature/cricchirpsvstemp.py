#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:03:14 2018

@author: kay
"""

#Cricket Chirps vs Temprature dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel("cricket.xls", sheetname=0)
#matrix of feature
cricket_chirps = dataset.iloc[:,1:2].values
#dependent vector
temp = dataset.iloc[:,-1].values

#test train split
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(cricket_chirps, temp, test_size=0.2)

#Model creation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain,ytrain)

ypred = regressor.predict(Xtest)

#model visualisation
plt.plot(Xtrain,regressor.predict(Xtrain),color='black')
plt.scatter(Xtrain,ytrain,color='red')
plt.scatter(Xtest,ytest,color='green')
plt.xlabel("Cricket Chirps")
plt.ylabel("Temprature")