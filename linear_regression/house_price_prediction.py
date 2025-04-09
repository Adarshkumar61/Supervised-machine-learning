#importing the required modules:
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

#importing house price dataset:

data = pd.read_csv('csv_files/housing.csv.zip')
#now we will check if there is any missing value in our data

data.isnull().sum() #this will check and tell us by combining all
#so there are some values that are missing 
#so we will fill up those missing values by ''

data = data.fillna(' ') # 

#now we will drop those columns which are not relevent and our 'Target'
# so column - Sea view not relevant
# our Target is :'median house price'
#and we will seperate our 'feature and 'Target

x = data.drop(['ocean proximity', 'median_house_value'], axis=1) #axis -1 bcz we are dropping column
y = data['medain_house_value']

#now we will split data into 2 parts:
# 1. training
# 2. testing
# we use train_test_split model to split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 2)
# test_size = 0.2 means we using 20 % data for testing and 80% data for training

#now we will call our model:

model = LinearRegression()
model.fit(x_train, y_train)

model_prediction  = model.predict(x_train)

accuracy_r2_score = metrics.r2_score(model_prediction, y_train) #this will check our model accuracy score
print(accuracy_r2_score)
accuracy_mean_absolute_error  = metrics.mean_absolute_error(model_prediction, y_train)
print(accuracy_mean_absolute_error)

#now we will create a predictive system

input_data = np.array([])
reshaping_array = input_data.reshape(1, -1)

prediction = model.predict(reshaping_array)
print(prediction) #this will give prediction

