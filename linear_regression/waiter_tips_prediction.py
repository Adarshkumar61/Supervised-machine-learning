import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('tips_data.csv')
print(data.head())

#    total_bill   tip     sex   smoker  day     time     size
# 0       16.99  1.01  Female     No    Sun     Dinner     2
# 1       10.34  1.66    Male     No    Sun     Dinner     3
# 2       21.01  3.50    Male     No    Sun     Dinner     3
# 3       23.68  3.31    Male     No    Sun     Dinner     2
# 4       24.59  3.61  Female     No    Sun     Dinner     4

data.isnull().sum()
#o missing values

# transforming categorical data into numerical:
data['sex'] = data['sex'].map({'Female': 0, 'Male': 1})
data['smoker'] = data['smoker'].map({'No': 0, 'Yes': 1})
data['day'] = data['day'].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data['time'] = data['time'].map({'Lunch': 0, 'Dinner': 1})

# split the data into features and target:
x = np.array(data['total_bill', 'sex', 'smoker', 'day', 'time', 'size']) # feature 
y = np.array(data['tip']) # target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)
model = LinearRegression()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))


y_pred = model.predict(x_test)

# Calculate Mean Squared Error and R² Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2) 
# Mean Squared Error: 1.201784681944687
# R² Score: 0.4645781929601126

# predicting a new data point:
# features = [total_bill, sex, smoker, day, time, size]
input_data = np.array([[30.0, 1, 0, 2, 1, 3]]) # features
pred = model.predict(input_data)
print('Predicted tip is: ', pred)