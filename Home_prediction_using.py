#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 15:30:57 2018

@author: Dmitry Gorochovsky
"""

#Import module for saving scikit-learn models
from sklearn.externals import joblib
import pandas as pd
import numpy as np

pd.DataFrame

tb = [{'CRIM' : 0.00632,
      'ZN' : 18.00,
      'INDUSTRY' : 2.310,
      'RIVER' : 0,
      'NOX' : 0.5380,
      'ROOMS' : 6.5750,
      'AGE' : 65.20,
      'DISTANCES' : 4.0900,
      'RAD' : 1,
      'TAX' : 296.0,
      'PTRATIO' : 15.30,
      'BLACK' : 396.90,
      'LSTAT' : 4.98}]

tb2 = [{'CRIM' : 0.02731,
      'ZN' : 26.00,
      'INDUSTRY' : 15.070,
      'RIVER' : 1,
      'NOX' : 0.4690,
      'ROOMS' : 15.4210,
      'AGE' : 10.20,
      'DISTANCES' : 4.961,
      'RAD' : 2,
      'TAX' : 300.0,
      'PTRATIO' : 17.30,
      'BLACK' : 500.90,
      'LSTAT' : 9.17}]


X_test = pd.DataFrame(tb)
X_test2 = pd.DataFrame(tb2)

#Load model from .pkl file
clf2 = joblib.load('home_price_regressor.pkl')
 
# Predict data set using loaded model
print('Price predicted on first sample (must be 24): ' + str(clf2.predict(X_test)))

print('Price predicted on my sample: ' + str(clf2.predict(X_test2)))