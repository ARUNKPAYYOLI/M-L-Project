#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[24]:


df = pd.read_csv('iris.csv')


# In[25]:


df.head()


# In[26]:


df.tail()


# In[27]:


df.info()


# In[28]:


df.drop('Id',axis=1,inplace=True)


# In[36]:


class_names = ['iris setosa','Iris-versicolor','Iris-virginica']


# In[37]:


iris = load_iris()
X = iris.data
y = iris.target


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[39]:


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# In[40]:


y_pred = clf.predict(X_test)


# In[41]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[42]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[43]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[ ]:





# In[ ]:




