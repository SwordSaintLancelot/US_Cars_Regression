# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:50:45 2021

@author: GHOST
"""

import pandas as pd
import numpy as np



# Data Exploration

cars_data = pd.read_csv(r'D:\GitHub\data\US_Cars_Regression\USA_cars_datasets.csv') # read dataset

dataset_shape = cars_data.shape

print(cars_data.head(10)) # Top 10 details in cars dataset

column_names = cars_data.columns # Name of the columns

for x in column_names:
    if 'NaN' in cars_data[x]:
        print(x)


