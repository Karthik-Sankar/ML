#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:34:26 2018

@author: kay
"""

#Best price vs list price slr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel("slr01.xls", sheetname=0)
#matrix of feature
listprice = dataset.iloc[:,1:2].values
#dependent vector
bestprice = dataset.iloc[:,-1].values

#test train split
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(listprice, bestprice, test_size=0.2)

#Model creation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain,ytrain)

ypred = regressor.predict(Xtest)

#model visualisation
plt.plot(Xtrain,regressor.predict(Xtrain),color='black')
plt.scatter(Xtrain,ytrain,color='red')
plt.scatter(Xtest,ytest,color='green')
plt.xlabel("List Price")
plt.ylabel("Best Price")