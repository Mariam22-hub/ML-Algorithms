#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Team
#Nora Mohamed Hussein 20201196
#Mariam mahomed elmoazen 20200528
#Heba Abdelwahab Sayed Abdelwahab 20201208
#Kholoud mohamed alkamkhli 20200846


# In[83]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression


# In[84]:


data = pd.read_csv("loan_old.csv")
data_new = pd.read_csv("loan_new.csv")

print ("\ndata\n",data.head(20))
print ("\ndata_new\n",data_new.head(20))
# data.shape[0]


# In[85]:


#drop loan_id ->loan_old
data = data.drop('Loan_ID', axis=1)
print("\ndata\n",data)

#drop loan_id ->loan_new
data_new = data_new.drop('Loan_ID', axis=1)
print("\ndata_new\n",data_new)


# In[86]:


#analysis
datanull= data.isnull().any(axis = 1) 
data[datanull]


# In[87]:


# check the type of each feature (categorical or numerical
feature_types = data.dtypes
print(feature_types)
values = data['Gender'].unique()
#print(values)


# In[88]:


# Isolating numerical columns
numerical_data = data.select_dtypes(include=[np.number])

# Calculating mean and standard deviation
means = np.mean(numerical_data, axis=0)
std_devs = np.std(numerical_data, axis=0)

print ("\n Data is not scaled\n")
print("Means of numerical features:\n", means)
print("Standard Deviations of numerical features:\n", std_devs)


# In[89]:


# sns.pairplot(data, hue="Max_Loan_Amount")
sns.pairplot(data)


# In[ ]:





# In[90]:


def RemoveMissingValue(data):
    datanull= data.isnull().any(axis = 1)  #remove row with missing value
    return data[~datanull]      


# In[91]:


#preprocssing remove rows with missing in loan_old , loan_new
data = RemoveMissingValue(data)  
data_new=RemoveMissingValue(data_new)  
print ("\ndata\n",data.head(20))
print ("\ndata_new\n",data_new.head(20))
values = data['Married'].unique()
# print(values)


# In[92]:


#shuffling data ->loan_old
shuffled_data = shuffle(data, random_state=42)
print('\nshuffled_data\n',shuffled_data)


# In[93]:


#separate features and target and split
x = shuffled_data.drop(columns=['Max_Loan_Amount','Loan_Status'])
y = shuffled_data[['Max_Loan_Amount','Loan_Status']]
x_predict = data_new
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
print("x_train ",x_train) , print("x_test",x_test),print("y_train ",y_train),print("y test ",y_test)
print(x_test.shape)


# In[94]:


#categorical features are encoded->loan_old 
x_train = x_train.replace({
    'Gender' :{'Male':0 , 'Female':1},
    'Married' : {'No':0,'Yes':1},
    'Education':{'Not Graduate':0, 'Graduate':1  },
    'Dependents':{'0':0,'1':1,'2':2,'3+':3  },
    'Property_Area':{'Rural':0 ,'Urban':1 ,'Semiurban':2}
})
x_predict = x_predict.replace({
    'Gender' :{'Male':0 , 'Female':1},
    'Married' : {'No':0,'Yes':1},
    'Education':{'Not Graduate':0, 'Graduate':1  },
    'Dependents':{'0':0,'1':1,'2':2,'3+':3  },
    'Property_Area':{'Rural':0 ,'Urban':1 ,'Semiurban':2}
})

x_test = x_test.replace({
    'Gender' :{'Male':0 , 'Female':1},
    'Married' : {'No':0,'Yes':1},
    'Education':{'Not Graduate':0, 'Graduate':1  },
    'Dependents':{'0':0,'1':1,'2':2,'3+':3  },
    'Property_Area':{'Rural':0 ,'Urban':1 ,'Semiurban':2}
})
print("\nx_train\n",x_train.head(20))
print("\nx_test\n",x_test.head(20))
print("\nx_predict\n",x_predict.head(20))


# In[95]:


#categorical target are encoded->loan_old 
y_train = y_train.replace({
    # 'Property_Area':{'Rural':0 ,'Urban':1 ,'Semiurban':2},
    'Loan_Status':{'N':0,'Y':1}    
})

y_test = y_test.replace({
    # 'Property_Area':{'Rural':0 ,'Urban':1 ,'Semiurban':2},
    'Loan_Status':{'N':0,'Y':1}    
})
print("\ny_train\n",y_train.head(20))
print("\ny_test\n",y_test.head(20))


# In[96]:


#categorical features are encoded->loan_new 
data_new = data_new.replace({
    'Gender' :{'Male':0 , 'Female':1},
    'Married' : {'No':0,'Yes':1},
    'Education':{'Not Graduate':0, 'Graduate':1  },
    'Dependents':{'0':0,'1':1,'2':2,'3+':3  },
    'Property_Area':{'Rural':0 ,'Urban':1 ,'Semiurban':2}
})
print(data_new.head(20))


# In[97]:


#numerical features are standardized ->loan_old
cols_to_standardize_data = ['Income','Coapplicant_Income','Loan_Tenor']
scaler = StandardScaler()
scaler.fit(x_train[cols_to_standardize_data])
x_train[cols_to_standardize_data] = scaler.transform(x_train[cols_to_standardize_data])  #transform fn turn dp to numpy so i change y to numpy also 
x_test[cols_to_standardize_data] = scaler.transform(x_test[cols_to_standardize_data])


x_train = x_train.to_numpy().reshape(-1,9)
x_test = x_test.to_numpy().reshape(-1,9)
print(y_test.shape)
y_train = y_train.to_numpy().reshape(-1,2)
y_test = y_test.to_numpy()
y_test_logistic = y_test[:,1]
x_test , x_train
x_predict = x_predict.to_numpy()
print("\n x_train\n",x_train)
print("\n x_test\n",x_test)
#loan_new ->numerical features are standardized
cols_to_standardize_data = ['Income','Coapplicant_Income','Loan_Tenor']
scaler.fit(data_new[cols_to_standardize_data])
data_new[cols_to_standardize_data] = scaler.transform(data_new[cols_to_standardize_data])  #transform fn turn dp to numpy so i change y to numpy also 
data_new = data_new.to_numpy().reshape(-1,9)
print("\ndata_new\n",data_new)


# In[98]:


y_train_linear = y_train[:,0]


# In[99]:


# linear regression model using sickit learn
linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train_linear)


# In[100]:


linear_regression_model.intercept_


# In[101]:


linear_regression_model.coef_


# In[102]:


# return the predicted y-values for each feature row in the x_test df
linear_model_predictions = linear_regression_model.predict(x_test)
linear_model_predictions


# In[103]:


# evaluating the model using R^2
y_test_linear = y_test[:,0]
linear_regression_model.score(x_test, y_test_linear)


# In[104]:


# predicting in the new loan data
linear_model_predictions_new = linear_regression_model.predict(data_new)
linear_model_predictions_new


# In[105]:


#define sigmoid function to calculate 1 / 1 + e^ (-z)
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


# In[106]:


# compute cost function for the logistic regression j(w,b) = 1/m*summation [loss(f w,b(x^(i))),y^(i)]
# loss function is cost per data point  = -y^(i)*log(fw,b(x^(i))) - (1-y^(i)) * log(1-fw,b(x^(i)))
def compute_cost_function(X,y,w,b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z)
        loss = -y[i] * np.log(f_wb_i) - (1-y[i]) * np.log(1-f_wb_i)
        cost+=loss
    cost = cost /m
    return cost


# In[107]:


w_tmp = np.array([1,1,1,1,1,1,1,1,1]) # **********delete one 1
print(w_tmp)
b_tmp = -3
print(compute_cost_function(x_train, y_train[:,1], w_tmp, b_tmp))


# In[108]:


def compute_gradient(X,y,w,b,lambda_):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        z = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z)
        error = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error * X[i,j]
        dj_db += error
    dj_dw = dj_dw/m
    dj_db = dj_db /m
    for i in range(n):
        dj_dw[i] = dj_dw[i]+(lambda_/m)*w[i]
    return dj_db , dj_dw


# In[109]:


# compute gradient descent for logistic regression 
def gradient_descent(X, y, w_initial, b_initial, alpha, num_iters):
    #store the cost at each iteration 
    old_j = []
    w = copy.deepcopy(w_initial)  #avoid modifying global w within function
    b = b_initial
    
    for i in range(num_iters):
        #get dj_dw , dj_dw 
        dj_db, dj_dw = compute_gradient(X, y, w, b,0.7)   

        # Update w, b
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      #get only the first 100000 value of the cost j
            old_j.append( compute_cost_function(X, y, w, b) )

        #print cost at every 20 interval to see the progress
        if i% math.ceil(num_iters / 20) == 0:
            print(f"Iteration {i:4d}: Cost {old_j[-1]}   ")
        
    return w, b, old_j           


# In[110]:


import copy
import math
# w_tmp = np.array([2.,3.])
# X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.zeros(9)# *******make it 9 instead of 10
new_array = y_train[:, 1]  # Remove the reshape operation
print(w_tmp)
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient(x_train, new_array, w_tmp, b_tmp, 0.7)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )
w,b,j = gradient_descent(x_train,new_array,w_tmp,b_tmp,0.1,1000)
print(w)
print(b)


# In[111]:


#predict function to predict if an entered user would get a loan or not
def predict_logistic(X, w, b): 
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(X[i],w) 
        z_wb += b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb>0.5 else 0
    return p


# In[112]:


tmp_p = predict_logistic(x_predict,w, b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')


# In[113]:


print(x_test.shape)
prediction = predict_logistic(x_test,w,b)
print(prediction.shape)
print(y_test.shape)
print('Train Accuracy: %f'%(np.mean(prediction == y_test_logistic) * 100))


# In[ ]:





# In[ ]:




