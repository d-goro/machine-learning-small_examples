# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
	
import numpy as np
import pandas as pd

#Import sampling helperPython
from sklearn.model_selection import train_test_split

#import preprocessing modules
from sklearn import preprocessing

#Import random forest model
from sklearn.ensemble import RandomForestRegressor

#Import cross-validation pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#Import evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

#Import module for saving scikit-learn models
from sklearn.externals import joblib




#Load wine data from remote URL
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=";")


#print(data.head())
#print(data.describe())

# separate target from training features
y = data.quality
X = data.drop('quality', axis=1)



#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)


#X_train_scaled = preprocessing.scale(X_train)


#print(X_train_scaled)
#print(X_train_scaled.mean(axis=0))
#print(X_train_scaled.std(axis=0))

#Fitting the Transformer API
scaler = preprocessing.StandardScaler().fit(X_train)

#Applying transformer to training data
X_train_scaled = scaler.transform(X_train)

#print(X_train_scaled.mean(axis=0))
#print("-----------------------------")
#print(X_train_scaled.std(axis=0))

#Pipeline with preprocessing and model
#Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

#print("-----------------------------")
#print(pipeline.get_params())

#when it's tuned through a pipeline, you'll need to prepend  randomforestregressor__

#Declare hyperparameters to tune
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}


#Tune model using cross-validation pipeline
#Sklearn cross-validation with pipeline. 
#GridSearchCV essentially performs cross-validation across the entire "grid" (all possible permutations) of hyperparameters.
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
# Fit and tune model
clf.fit(X_train, y_train)

#print(clf.best_params_)
#print("-----------------------------")

#Confirm model will be retrained
#print(clf.refit)

#Evaluate model pipeline on test data

#Predict a new set of data
y_pred = clf.predict(X_test)

#evaluate model performance. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
print('R^2 (coefficient of determination) regression score: ' + str(r2_score(y_test, y_pred)))

print('Mean squared error regression loss: ' + str(mean_squared_error(y_test, y_pred)))




#Save model to a .pkl file
joblib.dump(clf, 'rf_regressor.pkl')

#Load model from .pkl file
#clf2 = joblib.load('rf_regressor.pkl')
 
# Predict data set using loaded model
#clf2.predict(X_test)



