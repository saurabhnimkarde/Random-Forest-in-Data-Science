#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing                      


# In[3]:


company = pd.read_csv("Company_Data.csv")


# In[4]:


company.head()


# In[5]:


company.shape


# In[6]:


company.info()


# In[7]:


company.dtypes


# In[8]:


company['ShelveLoc'] = company['ShelveLoc'].astype('category')
company['Urban'] = company['Urban'].astype('category')
company['US'] = company['US'].astype('category')


# In[9]:


company.dtypes


# In[10]:


sales_mean = company.Sales.mean()
sales_mean


# In[11]:


company['High'] = company.Sales.map(lambda x: 1 if x > 8 else 0)


# In[12]:


company.High


# In[13]:


company


# In[14]:


# Encoding the categorical columns by using label encoder


# In[14]:


label_encoder = preprocessing.LabelEncoder()
company['ShelveLoc'] = label_encoder.fit_transform(company['ShelveLoc'])


# In[15]:


company['Urban'] = label_encoder.fit_transform(company['Urban'])


# In[16]:


company['US'] = label_encoder.fit_transform(company['US'])


# In[17]:


company


# In[18]:


X = company.iloc[:,1:11]
Y = company['High']


# In[19]:


X


# In[20]:


Y


# In[21]:


company['High'].unique()


# In[22]:


company.High.value_counts()


# In[24]:


#Inference: 236 values are there where the sales are less than 8 and 164 values are there where sales are greater than 8


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


# In[26]:


#Building a model using random forest


# In[24]:


rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[26]:


rf.fit(x_train,y_train)  
rf.estimators_  
rf.classes_ 
rf.n_classes_ 
rf.n_features_  

rf.n_outputs_ 

rf.oob_score_ 


# In[27]:


rf.predict(x_test)


# In[28]:


preds = rf.predict(x_test)
pd.Series(preds).value_counts()


# In[29]:


preds


# In[30]:


crosstable = pd.crosstab(y_test,preds)
crosstable


# In[31]:


np.mean(preds==y_test)


# In[32]:


print(classification_report(preds,y_test))


# In[ ]:




