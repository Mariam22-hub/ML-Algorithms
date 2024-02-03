#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score


# In[4]:


# Reload model from the file
loaded_bestmodel = joblib.load('best_model_.joblib')


# In[5]:


test_data = pd.read_csv('mnist_test.csv')


# In[6]:


Y_test_data = test_data['label']
X_test_data = test_data.iloc[:, 1:]
print(Y_test_data)
print(X_test_data)


# In[7]:


X_test_data = X_test_data/255.0
print(X_test_data)


# In[8]:


y_predictions = loaded_bestmodel.predict(X_test_data)

# Calculate and print the accuracy and confusion matrix
accuracy = accuracy_score(Y_test_data, y_predictions)
print("accuracy of best model on mnist_test.csv : ",accuracy)


# In[ ]:




