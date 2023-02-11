#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
from sklearn import datasets , linear_model
df = open('dataframe_.csv')
data = pd.read_csv(df)


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


input = data.input.to_list()
output = data.output.to_list()
input = np.array(input)
output = np.array(output)


# In[8]:


plt.scatter(data['input'] , data['output'])


# In[9]:


x = data[['input']]
y = data[['output']]


# In[10]:


x


# In[11]:


y


# In[13]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2)


# In[14]:


len(x_train)


# In[15]:


len(x_test)


# In[16]:


x_train


# In[41]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()


# In[38]:


for x in input:
    x = float(x)


# In[40]:


for y in output :
    y = float(y)


# In[42]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




