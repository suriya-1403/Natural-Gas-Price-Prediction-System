import numpy as np
import pandas as pd
import pickle

dataset=pd.read_csv(r'C:\Users\yeshwanth\Desktop\natural gas ibm project\daily_csv.csv')

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
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
y_pred=dtr.predict(x_test)
y_pred
from sklearn.metrics import r2_score
dtr_accuracy=r2_score(y_test,y_pred)
dtr_accuracy

pickle.dump(dtr,open('gas.pkl','wb'))
