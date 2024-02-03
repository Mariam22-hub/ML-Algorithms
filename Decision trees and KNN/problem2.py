#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[78]:


data = pandas.read_csv('diabetes.csv')
print(data.head())
# print(data.shape[0])


# ### Data preprocessing

# 1. Removing missing values

# In[79]:


def RemoveMissingValue(data):
    datanull= data.isnull().any(axis = 1)  #remove row with missing value
    return data[~datanull]  
data = RemoveMissingValue(data)  
print(data.shape[0])


# 2. splitting data

# In[80]:


x = data.drop('Outcome',axis=1)
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
print("x_train\n ",x_train.head()) , print("x_test\n",x_test.head()),print("y_train\n",y_train.head()),print("y test\n",y_test.head())
print(data.shape[0])
print(x_train.shape[0])
print(x_test.shape[0])


# 3. data normalization

# In[81]:


min = x_train.min(axis=0)
max = x_train.max(axis=0)
x_train = (x_train - min) / (max- min)
x_test = (x_test - min) / (max - min)

x_train = x_train.to_numpy().reshape((-1,8))
x_test = x_test.to_numpy().reshape((-1,8))
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


print("x_train\n",x_train)
print("x_test\n",x_test)  


# In[ ]:





# ### KNN Algorithm

# In[82]:


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)


# In[83]:


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # print(distances)
        
        # Get indices of the closest k neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Extract distances for the k nearest neighbors
        k_distances = [distances[i] for i in k_indices]

        # Compute weighted distances for the k nearest neighbors
        k_weighted_distances = [1/d if d != 0 else float('inf') for d in k_distances]
        
        # Count the occurrences of each label in the k nearest neighbors
        label_counts = np.bincount(k_nearest_labels)
        final_prediction = label_counts.argmax()

        # Check for ties
        if (label_counts == label_counts[final_prediction]).sum() > 1:
            # Tie detected, resolve using weighted distances
            sorted_indices_by_weight = np.argsort([-w for w in k_weighted_distances], kind='mergesort')
            
            # Update k_indices and k_nearest_labels based on the sorted weighted distances
            k_indices = [k_indices[i] for i in sorted_indices_by_weight]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            final_prediction = k_nearest_labels[0]

        return final_prediction


# In[84]:


knnalgorithm = KNN()
knnalgorithm.fit(x_train, y_train)
sum_Accuracy=0
cnt=0;

for k in range(1, 20):
    knnalgorithm.k = k;
    y_predicts = knnalgorithm.predict(x_test)
    correct_prediction = np.sum(y_predicts == y_test)
    Accuracy = correct_prediction / len(y_test)
    sum_Accuracy=sum_Accuracy+Accuracy
    cnt=cnt+1
    
    print("k value: ",knnalgorithm.k)
    print("Number of correctly classified instances: ",correct_prediction)
    print("Total number of instances: ",len(y_test))
    print("Accuracy: ",Accuracy*100)


print("average accuracy: ", sum_Accuracy/cnt)
print("average accuracy %100: ", (sum_Accuracy/cnt)*100)


# In[ ]:




