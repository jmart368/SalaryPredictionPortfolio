#!/usr/bin/env python
# coding: utf-8

# # Salary Predictions Based on Job Descriptions

# # Part 1 - DEFINE

# ### ---- 1 Define the problem ----

# The following is an examination of a set of job postings with their indicated salaries. Given this historical data and hiring trend, Human Resources would want to predict an employee's salary based on factors such as years of experience, job type, educational background, industry, and distance from metropolis 

# In[1]:


__author__ = "Jose Martinez"
__email__ = "jmart368@gmail.com"


# In[2]:


#import libraries
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


# ## Part 2 - DISCOVER

# ### ---- 2 Load the data ----

# In[3]:


# load the data into a Pandas dataframe
test_features_df = pd.read_csv('test_features.csv')
train_features_df = pd.read_csv('train_features.csv')
train_salaries_df = pd.read_csv('train_salaries.csv')


# In[4]:


# examine the 3 dataframes
test_features_df.info()


# In[5]:


# check for null values
test_features_df.isnull().sum()


# In[6]:


train_features_df.info()


# In[7]:


train_features_df.isnull().sum()


# In[8]:


train_salaries_df.info()


# In[9]:


train_salaries_df.isnull().sum()


# After confirming that no null values exist, we can therefore merge the train features and train salaries into one new dataframe which we can refer to as train_data_df.

# In[10]:


# merge test and train features on job id
train_data = pd.merge(train_features_df, train_salaries_df, on = 'jobId')
# check for null after merging 
train_data.isnull().sum()


# ### ---- 3 Clean the data ----

# In[11]:


# data dimensions
train_data.shape


# When we look at the dimensions of our data we can conclude that there are 1,000,000 rows representing the job profiles and 9 columns representing numerical/categorical data.

# In[12]:


# look for duplicate data
train_data.duplicated().sum()


# In[13]:


# search for invalid data (e.g. salaries <=0)
train_data[train_data['salary'] <= 0]


# When factoring the salaries where the salary <= 0, we can note that only 5 results rendered a 0 value.

# Since these 5 results are immaterial in comparison to 1,000,000 job profiles, it would be easier to drop these rows rather than guess the missing salaries. Guessing mean salaries would hinder the quality of our data set. Using a Mean or Median value as replacement would also overinflate the salary of a Junior level candidate as well as underinflate the salary of a Manager or VP.

# In[14]:


train_data[["jobType", "salary"]].groupby("jobType").describe()


# In[15]:


# drop missing data where salary is 0
train_data = train_data.loc[train_data['salary'] != 0]


# In[16]:


# confirm that 0 salary values were dropped
print(train_data.shape)
print(train_data[train_data['salary'] <= 0])


# ### ---- 4 Explore the data (EDA) ----

# In[17]:


# summarize numerical values 
train_data.describe(include = [np.number])


# In[18]:


# summarize categorical values
train_data.describe(include = ['object'])


# In[19]:


# Summarize the target variable - Salary
plt.figure(figsize = (18,5))
plt.subplot(1,2,1)
sns.boxplot(train_data.salary, color="Green")
plt.title("Salary Distribution")
plt.ylabel('Density')
plt.subplot(1,2,2)
sns.distplot(train_data.salary, bins=20, color="Green")
plt.title('Salary Distribution')
plt.show()


# Based on the Salary Distribution plots, we can see that the target variable is normally distributed with a right skewness due to some outliers beyond the $220 range. We can further confirm with the use of our outlier check function we do have some outliers whilet extracting the upper and lower range.

# In[20]:


def outlier_check(dfcol):
    """
    Function to extract IQR
    """
    sorted(dfcol)
    Q1, Q3 = np.percentile(dfcol , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return upper_range, lower_range

upper_range, lower_range = outlier_check(train_data['salary'])

print(f"The upper and lower salary are ${upper_range} and ${lower_range}.")


# In[21]:


def plot_features(df, col):
    """
    Function to plot each feature with the target variable
    """
    plt.figure (figsize = (16,8))
    plt.subplot(1,2,1)
    if df[col].dtype == 'int64' or col =='companyId':
        mean = df.groupby(col)['salary'].mean()
        std = df.groupby(col)['salary'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values-std.values, mean.values+std.values,                         alpha =0.1)
        plt.ylabel('Salaries')
    else :
        col_mean = df.groupby(col)['salary'].mean()
        df[col] = df[col].astype('category')
        levels = col_mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels, inplace=True)
        col_mean.sort_values().plot(kind='bar')
        plt.xticks(rotation=45)
        plt.xlabel(col)
        plt.ylabel('Averge salary by'+ ' ' + col)
        plt.subplot(1,2,2)
        sns.boxplot(x=col, y='salary', data=df)
        plt.xticks(rotation=45)    
        plt.ylabel('Salaries')
    plt.show()


# In[22]:


train_data_feat=['jobType', 'degree', 'major', 'industry', 'yearsExperience','milesFromMetropolis']
train_data_cat=['jobType', 'degree', 'major', 'industry'] 
train_data_numfeat=[]


# In[23]:


for features in train_data_feat:
    plot_features(train_data, features)


# A couple of observations to note from the above:
# * When we look at average salary by job type, c-suite postions end up making the most in salary.
# * Those who have at least bachelors degree or higher, tend to have a higher than average salary.
# * Business and Engineering majors tend to make more in average salaries in comparison toother majors.
# * The Finance and Oil Industry have higher average salaries than Education, Service, Auto, Health, and Web.
# * There is a postive correlation with having a higher salary and more years of experience.
# * There is a negative correlation with having a higher salary and living further way from a major city.

# In[24]:


def label_encode(df, col):
    """
    Function to convert each categorical variable by replacing and 
    using the average salary of that category
    """
    dict = {}
    cat_list = df[col].cat.categories.tolist()
    for cat in cat_list:
        dict[cat] = train_data[train_data[col] == cat]['salary'].mean()
    df[col] = train_data[col].map(dict)


# In[25]:


train_copy = train_data.copy()
for col in train_copy.columns:
    if train_copy[col].dtype.name == 'category':
        label_encode(train_copy, col)
        train_copy[col] = train_copy[col].astype('float')


# In[26]:


# create a correlation matrix to show the relationships between each variable
corr_matrix = train_copy.corr()
plt.figure(figsize=(14,12))
sns.heatmap(corr_matrix, annot=True, cmap='Blues_r')
plt.xticks(rotation=20)
plt.yticks(rotation=0)
plt.show()


# Based on the above correlation matrix we can note that there is a negative correlation with the following variables,
# * salary and milesFromMetropolis
# * industry and milesFromMetropolis
# * yearsExperience and major
# 
# All other variables are positively correlated with degree and major having the strongest correlation.

# ### ---- 5 Establish a baseline ----

# In[27]:


# select a reasonable metric (MSE in this case)
# create an extremely simple model and measure its efficacy
# e.g. use "average salary" for each industry as your model and then measure MSE
# during 5-fold cross-validation


# In[28]:


mse = mean_squared_error(train_copy['industry'], train_copy['salary'])
print(f'The baseline model Mean Squared Error is: {mse}')


# ### ---- 6 Hypothesize solution ----

# Brainstorm 3 models that you think may improve results over the baseline model based
# 
# **Linear Regression:** Basic regression model which can be used for any data set and size. It is also a straighforward model used for prediction.
# 
# **Random Forest:** A low bias model that is very fast and powerful to solve regression and classficiation problems. 
# 
# **Gradient Boosting:** A fast and high performanced based model that can create simple individual models by combining them into a new one.

# ## Part 3 - DEVELOP

# ### ---- 7 Engineer features  ----

# In[29]:


# create any new features needed to potentially enhance model
cat_variables = ['jobType', 'degree', 'major', 'industry'] 
num_variables = ['yearsExperience', 'milesFromMetropolis', 'salary']
feat_variables = ['jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']


# In[30]:


dummy_features_train = pd.get_dummies(train_data[cat_variables], drop_first=True)
train_df = pd.concat([dummy_features_train, train_data[num_variables]], axis=1)
train_df.head(5)


# In[31]:


# split the data into train and test sets with test size at 30%
X_train, X_test, Y_train, Y_test = train_test_split(train_df.iloc[:,:-1], train_df.salary, test_size=0.3)


# In[32]:


# confirm and observe the split
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ### ---- 8 Create models ----

# In[33]:


# create and tune the models that you brainstormed during part 2


# **8.1 Linear Regression**

# In[34]:


lr = LinearRegression()
lr.fit(X_train, Y_train)


# **8.2 Random Forest**

# In[35]:


rf = RandomForestRegressor(n_estimators=150,max_depth=25, max_features=25, random_state=0, min_samples_split=60)


# **8.3 Gradient Boosting Regressor**

# In[36]:


gr = GradientBoostingRegressor(n_estimators=160, max_depth=6, loss='ls', verbose=0 )


# ### ---- 9 Test models ----

# **9.1 Linear Regression Test**

# In[37]:


lr_score = lr.predict(X_test)
lr_mse = mean_squared_error(Y_test, lr_score) # calculate the mean square error of the linear model
lr_mse


# **9.2 Random Forest Test**

# In[38]:


rf_score = cross_val_score(rf, X_test, Y_test, cv=5, scoring="neg_mean_squared_error") 
rf_mse = -1.0 * np.mean(rf_score)
rf_mse


# **9.3 Gradient Boosting Regressor**

# In[39]:


gb_score = cross_val_score(gr, X_test, Y_test, cv=5, scoring="neg_mean_squared_error")
gb_mse = -1.0 * np.mean(gb_score)
gb_mse 


# ### ---- 10 Select best model  ----

# In[40]:


# select the model with the lowest error as your "production" model
models_MSE =pd.DataFrame(({'Models':['Linear regression','Random Forest','Gradient Boosting Regressor'],
               'Mean Squared Error':[lr_mse,rf_mse,gb_mse]}))
models_MSE


# Based on the lowest MSE, the best model is Gradient Boosting Regressor

# ## Part 4 - DEPLOY

# ### ---- 11 Automate pipeline ----

# In[41]:


# write script that trains model on entire training set, saves model to disk,
num_variables.remove('salary')


# In[42]:


dummy_features_test = pd.get_dummies(test_features_df[cat_variables], drop_first=True)
test_df = pd.concat([dummy_features_test, test_features_df[num_variables]], axis=1)


# In[43]:


test_df.head(6)


# In[44]:


gr.fit(train_df.iloc[:,:-1], train_df.salary)


# In[45]:


test_prediction = gr.predict(test_df) 
prediction = pd.DataFrame(test_prediction).rename(columns={0:'predicted_salary'})


# In[46]:


# score the "test" dataset
final_prediction = pd.concat([test_features_df['jobId'], prediction], axis=1)
final_prediction.head(6)


# ### ---- 12 Deploy solution ----

# In[47]:


# save your prediction to a csv file
final_prediction.to_csv("predicted_salaries.csv", index=False)


# ### ---- Feature Importances ----

# In[48]:


feature_importance = pd.DataFrame({'features': test_df.columns, 'importance': gr.feature_importances_})


# In[49]:


feature_importance.sort_values(by='importance', ascending=False, inplace=True)


# In[50]:


plt.figure(figsize=(18,10))
sns.barplot(x='importance', y='features', data=feature_importance)
plt.show();

