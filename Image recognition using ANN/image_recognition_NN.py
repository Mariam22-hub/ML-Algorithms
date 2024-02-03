#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Team
#Nora Mohamed Hussein 20201196
#Mariam mahomed elmoazen 20200528
#Heba Abdelwahab Sayed Abdelwahab 20201208
#Kholoud mohamed alkamkhli 20200846


# In[3]:


# !pip install tensorflow==2.3.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib


# In[4]:


data = pd.read_csv("mnist_train.csv")
data.head()


# In[5]:


unique_classes = data.nunique()
print(f"Number of unique classes: {unique_classes}")


# In[6]:


num_of_features = data.shape[0]
print(f"Number of features: {num_of_features}")


# In[7]:


missing = data.isnull().sum()
print(f"Number of missing values: {missing}")


# In[8]:


Y = data['label']
print(Y.head)


# In[9]:


X = data.iloc[:, 1:]
print(X.head)


# In[10]:


X = X/255.0
print(X)


# In[11]:


from PIL import Image
import matplotlib.pyplot as plt
def convert_to_image(row,width = 28 , height = 28):
    pixels = np.array(row).reshape(28,28)
    image = Image.fromarray(pixels.astype('uint8'))
    resized_image = image.resize((width, height))
    fig, axes = plt.subplots(1, 1, figsize=(3, 3))
    axes.imshow(pixels, cmap='gray')
    axes.set_title('Original Image')
    axes.axis('off')
for i in range(7):
    convert_to_image(X.iloc[i])


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score


# In[13]:


X_train, X_test , y_train , y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 42)
knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)


# In[14]:


print(y_pred[0:5])
print(y_test[0:5])


# In[15]:


print(f"accuracy score: {accuracy}")


# In[17]:


ranges_of_k  =  [3, 5, 9, 11]


# In[18]:


knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=ranges_of_k)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
# fitting the model for grid search
grid_search=grid.fit(X_train, y_train)


# In[19]:


print(grid_search.best_params_)


# In[20]:


best_knn_model = grid_search.best_estimator_
y_pred_ = best_knn_model.predict(X_test)
accuracy = accuracy_score(y_pred_, y_test)
print(f"Validation Accuracy: {accuracy:.4f}")


# In[21]:


test_data = pd.read_csv('mnist_test.csv')


# In[22]:


Y_test_data = test_data['label']
X_test_data = test_data.iloc[:, 1:]
print(Y_test_data)
print(X_test_data)


# In[23]:


X_test_data = X_test_data/255.0
print(X_test_data)


# In[24]:


y_test_predictions = best_knn_model.predict(X_test_data)


# In[25]:


accuracy = accuracy_score(y_test_predictions, Y_test_data)
print(f"Validation Accuracy: {accuracy:.4f}")
print(y_test_predictions)
print(Y_test_data)


# In[30]:


X_train, X_test , y_train , y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 42)


# In[31]:


X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)


# #### Architecture 1

# In[32]:


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))

# output layer
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))


# In[33]:


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[34]:


model.fit(X_train, y_train, epochs=5, validation_split=0.3)


# #### Architecture 2

# In[36]:


model2 = tf.keras.models.Sequential()

model2.add(tf.keras.layers.Dense(256, activation = 'relu'))
model2.add(tf.keras.layers.Dense(128, activation = 'relu'))
model2.add(tf.keras.layers.Dense(64, activation='relu'))

# output layer
model2.add(tf.keras.layers.Dense(10, activation = 'softmax'))

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model2.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[37]:


model2.fit(X_train, y_train, epochs=5, validation_split=0.3)


# In[38]:


ann_model1_accuracy=model.evaluate( X_test,y_test)[1]  #evaluate return loss - accuracy
ann_model2_accuracy=model2.evaluate( X_test,y_test)[1]

if(ann_model1_accuracy>ann_model2_accuracy):
  best_ann = model
else:
  best_ann = model2

print("ann_model1: ",ann_model1_accuracy)
print("ann model2: ",ann_model2_accuracy)

print(best_ann.evaluate( X_test,y_test)[1])


# In[40]:


best_ann_accu=best_ann.evaluate( X_test,y_test)[1]
y_predict = best_knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test,y_predict)

print("best_ann_accu: ",best_ann_accu)
print("knn_accu: ",knn_accuracy)

if(knn_accuracy > best_ann_accu):
  print("knn_accu is better than ann acuuracy: ",knn_accuracy)
  best_model = best_knn_model
else:
  print("ann_accu is better than knn acuuracy: ",best_ann_accu)

  best_model = best_ann

conf_matrix = confusion_matrix(y_test, best_model.predict(X_test))
print ("conf_matrix: \n",conf_matrix)


# In[41]:


joblib.dump(best_model, 'best_model_.joblib')


# In[ ]:




