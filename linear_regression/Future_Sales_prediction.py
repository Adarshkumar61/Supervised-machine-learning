import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('future_sales.csv')
print(data.head())
#       TV  Radio  Newspaper  Sales
# 0  230.1   37.8       69.2   22.1
# 1   44.5   39.3       45.1   10.4
# 2   17.2   45.9       69.3   12.0
# 3  151.5   41.3       58.5   16.5
# 4  180.8   10.8       58.4   17.9

data.isnull().sum()
# TV           0
# Radio        0
# Newspaper    0
# Sales        0
# so there is no missing values

x = data.drop('Sales', axis= 1) # features
y = data['Sales'] # target

