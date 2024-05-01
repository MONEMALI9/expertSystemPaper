#!/usr/bin/env python
# coding: utf-8

# # Project: case study in Expert system using the age and minutes played feature. - [we use data downloaded from FBref on players in La Liga.]
# 
# ## Table of Contents
# <ul>
#     <li>
#         <a href="#intro">
#             Introduction
#         </a>
#     </li>
#     <li>
#         <a href="#wrangling">
#             Data Wrangling
#             <ol>
#             <li>
#                 <a href="#Gathering">Gathering Data</a>
#             </li>
#             <li>
#                 <a href="#Assessing">Assessing Data</a>
#             </li>
#             <li>
#                 <a href="#Cleaning">Cleaning Data</a>
#             </li>
#             </ol>
#         </a>
#     </li>
#     <li>
#         <a href="#eda">
#             Exploratory Data Analysis
#         </a>
#     </li>
#     <li>
#         <a href="#ES">
#             Expert System
#         </a>
#     </li>
#     <li>
#         <a href="#mlm">
#             Machine Learning Model
#         </a>
#     </li>
#     <li>
#         <a href="#conclusions">
#             Conclusions
#         </a>
#     </li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# > **Tip**: In this section of the report, provide a brief introduction to the dataset you've selected/downloaded for analysis. Read through the description available on the homepage-links present [here](https://fbref.com/en/comps/8/passing/Champions-League-Stats). List all column names in each table, and their significance. In case of multiple tables, describe the relationship between tables. 
# 

# In[1]:


#ipython kernel install --user --name=venv
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**

# Data Wrangling which include :
# 
#     1.Gathering Data
#     2.Assessing Data
#     3.cleaning Data

# <a id='Gathering'></a>
# ## Gathering Data
# In this example we use data downloaded from [FBref](https://fbref.com/en/comps/12/2021-2022/stats/2021-2022-La-Liga-Stats) on players in La Liga.
# We just use the age and minutes played columns.
# And we only take the first 15 observations, to help visualise the process.
# Download [playerstats.csv](https://github.com/soccermatics/Soccermatics/blob/main/course/lessons/lesson2/playerstats.csv)
# your working directory.

# In[2]:


laliga_df=pd.read_csv("playerstats.csv",delimiter=',')


# In[3]:


laliga_df.head()


# #### Data Dictionary
# 
# 01 - id     : player id.
# 
# 02 - name   : player name.
# 
# 03 - Nation : player nationality.
# 
# 04 - pos    : the player position.
# 
# 05 - Squad  : the club where player play in.
# 
# 06 - Age    : How old is the player.
#    
# 07 - Born   : the year Where player born in.
# 
# 08 - MP     : the number of matches that player play
# 
# 09 - Starts : the number of matches that player start playing in it.
# 
# 10 - Min    : the number of matches that player play in all matches
# 
# 11 - 90s    : the number of matches that player play if he play from the beginning of the match to the end
# 
# 12 - Gls    : the number of goals that player acheives in all matches
# 
# 13 - Ast    : the number of goals that player acheives in all matches
# 
# 14 - CrdY   : the number of yellow cards that player took in all matches
# 
# 15 - CrdR   : the number of red cards that player took in all matches

# <a id='Assessing'></a>
# ## Assessing Data 
# We assessing our data using some function like : shape , ndim , dtypes , size  , info() , nunique() , isnull()

# In[4]:


# return number of columns and number of row
laliga_df.shape


# In[5]:


#return number of dimensions of data
laliga_df.ndim


# In[6]:


#return types of each column
laliga_df.dtypes


# In[7]:


#return number of unique value
laliga_df.nunique()


# In[8]:


# return which value is nul or not for each columns in DataSet 
laliga_df.isnull().any()


# In[9]:


#return number of columns has a null value
laliga_df.isnull().any().sum()


# In[10]:


#return number of columns has a null value
laliga_df.isnull().sum()


# In[11]:


laliga_df.info()


# <a id='Cleaning'></a>
# ## Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
# 
#     1.duplicated data
#     
#     2.missing value
#     
#     3.incorrect datatype

# ### duplicicated data

# In[12]:


#check for duplicated data
sum(laliga_df.duplicated())


# ### missing value

# In[13]:


laliga_df.dropna(inplace = True)


# ### incorrect datatype

# In[14]:


laliga_df[['Age','Born']] = laliga_df[['Age','Born']].astype('Int64')


# In[15]:


laliga_df.info()


# In[16]:


laliga_df.to_csv("playerstatsProcessed.csv", encoding='utf-8',index=False)


# In[17]:


laliga_df_processed = pd.read_csv("playerstatsProcessed.csv",delimiter=',')


# In[18]:


laliga_df_processed.head()


# In[19]:


laliga_df_processed.info()


# <a id='eda'></a>
# ## Exploratory Data Analysis

# In[20]:


#return statistical descriptive of dataset for each column
laliga_df_processed[['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'CrdY', 'CrdR']].describe()


# In[21]:


from statmeasures import statisticsMeasures


# In[22]:


statisticsMeasures('std')


# <a id='ES'></a>
# ## Expert System
# 
# Sure, let's start by defining the knowledge base, facts, rules, and the inference engine for an expert system using the data downloaded from FBref on players in La Liga. We'll focus on using the age and minutes played columns from the playerstatsProcessed.csv file

# Here's an outline of how we can structure these components:
# 
# 1. Knowledge Base:
#    
#     1. Read the data from the playerstats.csv file.
#     2. Extract the age and minutes played columns for the first 15 observations. 
# 
# 2. Facts:
# 
#    1. Define facts based on the extracted data.
# 
# 3. Rules:
#     
#     1. Define rules based on the relationships between age and minutes played.
# 
# 4. Inference Engine:
# 
#     1. Apply rules to facts to derive conclusions.

# In this implementation:
# 
# 1. We read the data from the playerstats.csv file and extract the age and minutes played columns for the first 15 observations.
# 
# 2. We define a simple rule stating that if a player's age is less than 25 and they have played more than 1000 minutes, they are considered a key player.
# 
# 3. The inference engine applies this rule to the facts derived from the knowledge base and generates conclusions.

# In[23]:


import expertsys
from expertsys import *
file_path = "playerstatsProcessed.csv"
data = read_data(file_path)
conclusions = infer(data)
for conclusion in conclusions:
    print(conclusion)


# In[25]:


# Apply the classification function to each row in the knowledge base
data['Classification'] = data.apply(classify_players, axis=1)

# Display the updated knowledge base with classifications
print(data[['name', 'Age', 'Min', 'Classification']])


# <a id='mlm'></a>
# ## Machine Learning Model

# In[26]:


# Splitting the data into features (X) and target variable (y)
X = laliga_df_processed[['Age']]
y = laliga_df_processed['Min']


# Splitting the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Plot the training data and the linear regression line
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model.predict(X_train), color='red', label='Linear regression line')
plt.xlabel('Age')
plt.ylabel('Minutes Played')
plt.title('Linear Regression: Age vs Minutes Played (Training Data)')
plt.legend()
plt.show()

# Plot the testing data and the linear regression line
plt.scatter(X_test, y_test, color='blue', label='Testing data')
plt.plot(X_test, y_pred, color='red', label='Linear regression line')
plt.xlabel('Age')
plt.ylabel('Minutes Played')
plt.title('Linear Regression: Age vs Minutes Played (Testing Data)')
plt.legend()
plt.show()


# ### checking fitting for this code
# To check the goodness of fit for the linear regression model.
# you can use various metrics such as R-squared, mean squared error (MSE), or mean absolute error (MAE). Here's how you can calculate and print these metrics in Python

# In[27]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Evaluate training set performance
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)

print("Training Set Performance:")
print("R-squared:", r2_train)
print("Mean Squared Error:", mse_train)
print("Mean Absolute Error:", mae_train)

# Evaluate testing set performance
r2_test = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)

print("\nTesting Set Performance:")
print("R-squared:", r2_test)
print("Mean Squared Error:", mse_test)
print("Mean Absolute Error:", mae_test)

# Visual inspection
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X_train, model.predict(X_train), color='red', label='Linear regression line')
plt.xlabel('Age')
plt.ylabel('Minutes Played')
plt.title('Linear Regression: Age vs Minutes Played')
plt.legend()
plt.show()


# In[ ]:





# ## Fitting the model
# We are going to begin by doing a  straight line linear regression
#  .. math::
# 
#     y = b_0 + b_1 x
# 
# A straight line relationship between minutes played and age.
# 
# 

# In[28]:


num_obs=20
minutes_model = pd.DataFrame()
minutes_model = minutes_model.assign(minutes=laliga_df['Min'][0:num_obs])
minutes_model = minutes_model.assign(age=laliga_df['Age'][0:num_obs])

# Make an age squared column so we can fir polynomial model.
minutes_model = minutes_model.assign(age_squared=np.power(laliga_df['Age'][0:num_obs],2))


# ## Plotting the data
# Start by plotting the data.
# 
# 

# In[29]:


fig,ax=plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker= '.', markersize= 10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15,40))
plt.ylim((0,3000))
plt.show()


# In[30]:


model_fit=smf.ols(formula='minutes  ~ age   ', data=minutes_model).fit()
print(model_fit.summary())        
b=model_fit.params


# Comparing the fit 
#  ----------------------------
# We now use the fit to plot a line through the data.
#  .. math::
# 
#     y = b_0 + b_1 x
# 
# where the parameters are estimated from the model fit.
# 
# 

# In[31]:


#First plot the data as previously
fig,ax=plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker= '.', markersize= 10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15,40))
plt.ylim((0,3000))

#Now create the line through the data
x=np.arange(40,step=1)
y= np.mean(minutes_model['minutes'])*np.ones(40)
ax.plot(x, y, color='black')

#Show distances to line for each point
for i,a in enumerate(minutes_model['age']):
    ax.plot([a,a],[minutes_model['minutes'][i], np.mean(minutes_model['minutes']) ], color='red')
plt.show()


# A model including squared terms
#  ----------------------------
# We now fit the quadratic model
#  .. math::
# 
#     y = b_0 + b_1 x + b_2 x^2
# 
# estimating the parameters from the data.
# 
# 

# In[32]:


# First fit the model
model_fit=smf.ols(formula='minutes  ~ age + age_squared  ', data=minutes_model).fit()
print(model_fit.summary())        
b=model_fit.params

# Compare the fit 
fig,ax=plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker= '.', markersize= 10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15,40))
plt.ylim((0,3000))
x=np.arange(40,step=1)
y= b[0] + b[1]*x + b[2]*x*x
ax.plot(x, y, color='black')

for i,a in enumerate(minutes_model['age']):
    ax.plot([a,a],[minutes_model['minutes'][i], b[0] + b[1]*a + b[2]*a*a], color='red')
plt.show()


# In[35]:


get_ipython().system(' jupyter nbconvert --to html to expertSystem.ipynb')


# In[36]:


get_ipython().system(' jupyter nbconvert --to python to expertSystem.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




