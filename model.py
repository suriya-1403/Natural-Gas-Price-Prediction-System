#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:01:20 2021

@author: fasil
"""

import pandas as pd
import joblib


dataset=pd.read_csv(r'/Users/fasil/Desktop/git/Natural-Gas-Price-Prediction-System-main/data/naturalgas.csv')

dataset['year']=pd.DatetimeIndex(dataset['Date']).year
dataset['month']=pd.DatetimeIndex(dataset['Date']).month
dataset['day']=pd.DatetimeIndex(dataset['Date']).day
dataset.drop('Date',axis=1,inplace=True)
dataset.isnull().any()
dataset['Price'].fillna(dataset['Price'].mean(),inplace=True)

x=dataset.iloc[:,1:4].values
y=dataset.iloc[:,0:1].values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
DecisionTreeRegressor()
df_grid = GridSearchCV(DecisionTreeRegressor(),
                  param_grid = {'criterion':['mse', 'friedman_mse', 'mae', 'poisson'],
                                'splitter': ['best', 'random'],
                                'max_depth': range(1, 11),
                                'min_samples_split': range(10, 60, 10),
                                },
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')

df_grid.fit(x_train, y_train)
df=DecisionTreeRegressor(criterion='mse',max_depth=10,min_samples_split=10,splitter='best')
df.fit(x_train,y_train)
y_pred=df.predict(x_test)
y_pred
from sklearn.metrics import r2_score
df_accuracy=r2_score(y_test,y_pred)
df_accuracy

joblib.dump(df,"models/DecisionTreeRegressor.save")

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=1, random_state=10)
rfr.fit(x_train,y_train)
y_pred_rfr=rfr.predict(x_test)
y_pred_rfr
accur_rfr=r2_score(y_test,y_pred_rfr)
accur_rfr

joblib.dump(rfr,"models/RandomForestRegressor.save")