#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Read Data

# In[3]:


data = pd.read_csv('/Users/content/Linear_regression/Salary_dataset.csv')
data


# ### Remove unnecessary columns

# In[4]:


del data['Unnamed: 0']
data


# ### Clean Data

# In[5]:


data.head()


# In[6]:


data.isnull().sum()


# In[7]:


data.shape


# In[8]:


data.dtypes


# In[9]:


data = data.drop_duplicates()
data.shape


# No duplicate data

# In[10]:


data.describe()


# In[11]:


#outliers in feature variable
iqr = data.YearsExperience.quantile(0.75) - data.YearsExperience.quantile(0.25)
iqr


# In[14]:


upper_threshold = data.YearsExperience.quantile(0.75) + (1.5 * iqr)
lower_threshold = data.YearsExperience.quantile(0.25) - (1.5 * iqr)
upper_threshold , lower_threshold


# values > than upper_threshold --> outliers
# values < than lower_threshold --> outliers
# No outliers in data

# ### EDA

# Plot data to check Linear Relation

# In[15]:


data.plot(x = 'YearsExperience' , y = 'Salary' , style = 'o')
plt.title(' Years of Experience vs Salary')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()


# There is Linear Relation between feature and target so, no transformation of feature required.

# Deriving Correlation to check Linear Relation

# In[16]:


data.corr()


# Strong positive correlation, there is linear relation between feature and target variable so no transformation required.

# ### Splitting data for training and testing

# In[17]:


x = data.loc[: , ['YearsExperience']].values
y = data.loc[: , 'Salary'].values

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 45)


# In[18]:


x,y


# In[19]:


y_test


# In[20]:


x_train.shape , x_test.shape


# In[21]:


y_train.shape , y_test.shape


# ### Model for algorathim salary = m * YearsExperience + c

# In[22]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #predicted Salary = m * YearsExperience + c
regressor.fit(x_train , y_train)


# In[23]:


print(regressor.intercept_) # c


# In[24]:


print(regressor.coef_) #slope m


# In[25]:


regressor.predict([[5.5]])# predicting salary for 5.5 years of experience


# In[26]:


y_pred = regressor.predict(x_test) #predictions for feature testing data
y_pred


# In[27]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# ### Evaluation Metrics

# In[28]:


from sklearn import metrics 
print('R2- SCORE:', metrics.r2_score(y_test,y_pred))


# Thus, the R-Squared Score of 94% suggests that the model developed is a very accurate model to derive Salary based on years of experience
