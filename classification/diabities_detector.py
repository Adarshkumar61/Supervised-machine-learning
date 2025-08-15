# importing required modules:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('C:\Users\adars\Adarsh\data.csv')

# check if there is any missing value present or not:
data.isnull().sum()

# we are gonna seperate feature and column: 

x = data.drop(columns= 'outcome', axis= 1) #feature
y = data['outcome']  #target

# splitting data into training and testing test: 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify= y, random_state=2)

# now we call our model:
linear_model = LogisticRegression() #our model

linear_model.fit(x_train, y_train) #training the data

# prediction on x_train and x_test:

predict = linear_model.predict(x_train)
predict_x_test = linear_model.predict(x_test) #now we will check the accuracy of our model:

# on x_train:
accuracy_of_x_train = accuracy_score(y_train, predict)  

# on x_test:
accuracy_of_x_test = accuracy_score(y_test, predict_x_test)


#creating a input parameter: 

user_input = ([]) # values
in_numpy = np.asarray(user_input)

reshaped_model = in_numpy.reshape(1, -1)
print(reshaped_model) # it will give prediction 
