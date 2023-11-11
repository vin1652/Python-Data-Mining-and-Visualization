# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:17:30 2023

@author: vinay
"""
## Import data 
import pandas as pd
import numpy as np
#TASK1
toyota_df = pd.read_csv("C:/Users/vinay/OneDrive/Documents/.spyder-py3/ToyotaCorolla.csv")
toyota_df=toyota_df.drop(columns=['Id', 'Model'])
toyota_df.columns
X=toyota_df.drop(columns=['Price'])
y=toyota_df['Price']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns) # transform into a dataframe and add column names

#TASK2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 5)


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Run ridge regression with penalty equals to 1
ridge = Ridge(alpha=1)
ridge_model = ridge.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 1 = "+str(round(ridge_mse,0)))

#TASK3 
# Building isolation forest model
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, contamination=.05)
pred = iforest.fit_predict(toyota_df)
score = iforest.decision_function(toyota_df)
# Extracting anomalies
from numpy import where
anomaly_index = where(pred==-1)[0]

#TASK 4 -REMOVING anomalies from dataset 
X_train_clean = X_train.drop(index=anomaly_index, errors='ignore')
y_train_clean = y_train.drop(index=anomaly_index, errors='ignore')
X_test_clean = X_test.drop(index=anomaly_index, errors='ignore')
y_test_clean = y_test.drop(index=anomaly_index, errors='ignore')
#TASK 5
# Run ridge regression with penalty equals to 1
ridge1 = Ridge(alpha=1)
ridge_model1 = ridge1.fit(X_train_clean,y_train_clean)

# Generate the prediction value from the test data
y1_test_pred = ridge_model1.predict(X_test_clean)

# Calculate the MSE
ridge_mse1 = mean_squared_error(y_test_clean, y1_test_pred)
print("Test MSE using ridge with penalty of 1 = "+str(round(ridge_mse1,0)))


