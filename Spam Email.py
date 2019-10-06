#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[2]:


data= pd.read_csv("D:\ML\Project\Spam Email\spam.csv")
data.head()


# In[3]:


data.describe()


# In[4]:


#Splitting the dataset
from sklearn.model_selection import train_test_split

x= data.iloc[:, 1]
y= data.iloc[:, 0]

x_train, x_test, y_train, y_test= train_test_split(x, y, random_state= 0, test_size= 0.2)


# In[5]:


#Converting string into numbers in data
cv= CountVectorizer()
features= cv.fit_transform(x_train)
features_test= cv.fit_transform(x_test)


# In[6]:


#Preparing model
model= svm.SVC()
model.fit(features, y_train)


# In[15]:


features_test= cv.fit_transform(x_test)
features_test


# In[18]:


model.score(features_test , y_test)


# In[ ]:




