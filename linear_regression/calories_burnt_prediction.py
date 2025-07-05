import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# import our dataset:
# there are 2 dataset:
# 1.exercise dataset
# 2.calories dataset

exercise = pd.read_csv('csv_files/exercise.csv')
calories = pd.read_csv('csv_files/calories.csv')

# combine both dataset:
combine = pd.concat([exercise, calories['Calories']], axis=1) 
#concating along the columns

combine.isnull().sum()
#so there are no missing values in dataset

combine.drop('User_ID', axis=1, inplace= True)

combine.replace({'Gender': {'male': 1, 'female': 0}}, inplace= True) 

#so we are seperating feature and our target:

x = combine.drop('Calories', axis=1) #feature
y = combine['Calories'] #target
 
#sending data into traning and testing :

x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size= 0.2, random_state=2)

scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

#we are using model:

model = GradientBoostingRegressor()
model.fit(x_train, y_train) 


prediction_on_x_train = model.predict(x_train)
prediction_on_x_test = model.predict(x_test)

#accuracy of model:

# using r2 score:
scorer2_x_train = r2_score(y_train, prediction_on_x_train) #accuracy of x train

scorer2_x_test = r2_score(y_test, prediction_on_x_test) #accuracy of x train

# using mean_absolute_error:
score1_mea = mean_absolute_error(y_train, prediction_on_x_train) #accuracy of x train
score2_mea = mean_absolute_error(y_test, prediction_on_x_test) #accuracy of x train


user_input = ([0, 38, 160.0, 81.0, 29.0,105.0, 41.8])
user_input_numpy = np.asarray(user_input)
reshaped = np.reshape(user_input_numpy)
prediction = model.predict(reshaped)
print(prediction) #it tells how many calories burnt based on some info.