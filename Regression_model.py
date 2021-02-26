# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:50:45 2021

@author: GHOST
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Data Exploration

cars_data = pd.read_csv(r'D:\GitHub\data\US_Cars_Regression\USA_cars_datasets.csv') # read dataset

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
sns.displot(cars_data['price'], kind = 'hist').set_titles('Price Distribution')

for name in column_names:
    print(name)
    print(cars_data[name].value_counts().size)
    
for name in column_names:
    plt.figure(figsize = (7,7))
    sns.barplot(x=name, y = cars_data['price'], data= cars_data)