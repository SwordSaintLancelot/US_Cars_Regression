# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:50:45 2021

@author: GHOST
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# import os

# os.chdir('D:\Github')
# Data Exploration

cars_data = pd.read_csv(r'.\data\USA_cars_datasets.csv') # read dataset

dataset_shape = cars_data.shape

print(cars_data.head(10)) # Top 10 details in cars dataset

column_names = cars_data.columns # Name of the columns

'''
After reading through the data, we can figure out that
the columns with names below are not relevant for finding out the price of a car:
1. Unnamed: 0
2. vin 
3. lot
4. country
'''
cars_data = cars_data.drop(columns = ['Unnamed: 0', 'vin', 'lot', 'country'])

column_names = cars_data.columns  # Updated column names

# Check if the dataset has some missing values

cars_data.isna().sum().sum()

# Create figures for Price distributions
plt.figure(figsize = (5,5))
sns.displot(cars_data['price'], kind = 'kde').set_titles('Price Distribution')
# sns.displot(cars_data['price'], kind = 'hist').set_titles('Price Distribution')

for name in column_names:
    print(name)
    print(cars_data[name].value_counts())
    print(cars_data[name].value_counts().size)
    
for name in column_names:
    plt.figure(figsize = (15,15))
    plt.title(name)
    # Plotting bar graph
    cars_data[name].value_counts().plot(kind='bar', x=name)
    # Plotting the box plot to figure out the average price of the cars.
    sns.boxplot(x=name, y = cars_data['price'], data= cars_data)
    
    
'''
Claculating the best features to create the ML models.
We will visualize and calculate the importance of each feature.
Along with this, we will make the changes to the dataset on the basis of 
the importance calculated, i.e. group various entities into single 
and eliminate the outliers.
'''
    
# Combining the brands which have the least number of cars.
# After looking at the pictorial representation earlier we ccan figure out that there are just a few number of cars
# which have the most number of cars sold.
other_brands = cars_data.brand.value_counts().index[15:]
cars_data.brand = cars_data.brand.replace(other_brands, 'other_brands')

# Similar to models, the number of models sold for each brand vary.
# If we plot the price vs model for a single brand we can figure out that models also play a significant role.

plt.figure(figsize=(15,15))
plt.title('Price comparison of various model of Nissan')
sns.boxplot(x = 'model', y = 'price', data = cars_data[cars_data.brand == 'nissan'])

other_models = cars_data.model.value_counts().index[60:]
cars_data.model = cars_data.model.replace(other_models, 'other_models')

# Similarly we will work through all the features.

other_colors = cars_data.color.value_counts().index[15:]
cars_data.color = cars_data.color.replace(other_colors, 'other_colors')

# Calculating the age of car
cars_data.year = 2021- cars_data.year



# Removing the outliers
cars_data = cars_data[cars_data.price<50000]

# Plotting dataset again

column_names = cars_data.columns
for name in column_names:
    plt.figure(figsize = (15,15))
    plt.title(name)
    # Plotting bar graph
    cars_data[name].value_counts().plot(kind='bar', x=name)
    # Plotting the box plot to figure out the average price of the cars.
    sns.boxplot(x=name, y = cars_data['price'], data= cars_data)

# Encoding the label variables
# the labels in the dataset are brand names, title status, model name
encoded_labels = pd.get_dummies(cars_data[['brand', 'model', 'title_status']],drop_first=True)
    
# Normalization
numerical_features = cars_data[['mileage','year']]

scaler = MinMaxScaler()
scaled_numerical_features = scaler.fit_transform(numerical_features)


data = np.concatenate((encoded_labels.values,scaled_numerical_features), axis = 1)

X = data[:,:data.shape[1]-1]
Y = np.array(cars_data['price'])


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=10)


model1 = RandomForestRegressor(n_estimators=500,verbose = 1, max_depth=40, n_jobs=5)

model1.fit(x_train,y_train)
scoring1 = model1.score(x_test,y_test)


model2 = GradientBoostingRegressor(n_estimators=500, verbose = 1, random_state=15, max_depth=3)

model2.fit(x_train,y_train)

scoring2 = model2.score(x_test,y_test)


model3 = LinearRegression(n_jobs = 4)
model3.fit(x_train,y_train)

scoring3 = model3.score(x_test,y_test)


pred = model2.predict(x_test)
