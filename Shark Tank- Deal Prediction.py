#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np


# In[40]:


st = pd.read_csv("SharkTank.csv")
st.columns


# In[41]:


st_cleaned=st.drop(columns=['Season Number', 'Season Start', 'Season End', 'Episode Number',
       'Pitch Number', 'Original Air Date', 'Startup Name', 'Entrepreneur Names','Total Deal Amount', 'Total Deal Equity', 'Deal Valuation',
       'Number of sharks in deal', 'Investment Amount Per Shark',
       'Equity Per Shark', 'Royalty Deal', 'Loan',
       'Barbara Corcoran Investment Amount',
       'Barbara Corcoran Investment Equity', 'Mark Cuban Investment Amount',
       'Mark Cuban Investment Equity', 'Lori Greiner Investment Amount',
       'Lori Greiner Investment Equity', 'Robert Herjavec Investment Amount',
       'Robert Herjavec Investment Equity', 'Daymond John Investment Amount',
       'Daymond John Investment Equity', 'Kevin O Leary Investment Amount',
       'Kevin O Leary Investment Equity','Business Description'])


# In[19]:


st_cleaned.columns


# In[20]:


X=st_cleaned.drop(columns=['Got Deal'])
y=st_cleaned['Got Deal']


# In[21]:


X = pd.get_dummies(data = X, columns = ['Industry','Pitchers Gender',
       'Pitchers City', 'Pitchers State', 'Pitchers Average Age'])


# In[22]:


# Standardize the dataset
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
scaled_X = standardizer.fit_transform(X)




# In[23]:


# Load libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score



# In[25]:


#Choosing k
k_values = [50,100,150,200]


for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k,p=2)
    scores = cross_val_score(knn, scaled_X, y, cv=5, scoring='accuracy')
    print(f'Accuracy score using k-NN with {k} neighbors: {scores.mean():.3f}')



# In[36]:


# Run LASSO with alpha=0.05
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.025) # you can control the number of predictors through alpha
model = ls.fit(scaled_X,y)

df=pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])


# In[37]:


# Filter and display only rows with non-zero coefficients
print(df[df['coefficient'] != 0])



# In[38]:


selected_columns = df[predictor].tolist()
indices = [X.columns.get_loc(col) for col in selected_columns]
X_lasso = X.iloc[:, indices]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_lasso, y, test_size = 0.3, random_state = 5)

# Standardize the dataset
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
scaled_X_train = standardizer.fit_transform(X_train)
scaled_X_test = standardizer.transform(X_test)

# Build a model with k = 200 and using euclidean distance function
knn = KNeighborsClassifier(n_neighbors=200,p=2)
model2_start=timeit.default_timer()
model2 = knn.fit(scaled_X_train,y_train)
model2_stop=timeit.default_timer()
# Using the model to predict the results based on the test dataset
y_test_pred = model2.predict(scaled_X_test)
# Get accuracy score
model2_accuracy=accuracy_score(y_test, y_test_pred)
model2_precision = precision_score(y_test, y_test_pred)  
model2_recall = recall_score(y_test, y_test_pred)  
model2_time=model2_stop-model2_start


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load the dataset
# df = pd.read_csv("your_dataset_path.csv")

# Assuming you've loaded the data into df
df = your_dataframe

# Data Preprocessing: 
# Convert categorical columns to numerical
# This is just an example; you'd have to consider each column's data type and meaning
label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data into training and test sets
X = df.drop('Got Deal', axis=1)  # Assuming 'Got Deal' is the target variable
y = df['Got Deal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN with GridSearchCV
knn_params = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=5)
knn_grid.fit(X_train_scaled, y_train)

print(f"Best parameters for KNN: {knn_grid.best_params_}")

# GradientBoostingClassifier with GridSearchCV
gbc_params = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5]
}
gbc = GradientBoostingClassifier()
gbc_grid = GridSearchCV(gbc, gbc_params, cv=5)
gbc_grid.fit(X_train, y_train)

print(f"Best parameters for GradientBoostingClassifier: {gbc_grid.best_params_}")

# You can now evaluate the models using X_test and y_test or any other metric of choice.

