#importing required modules
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#loading the dataset
data = pd.read_csv('csv_files/car data.csv')

#features are:  Car_Name, Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
# Target: Selling_Price

#now we will check if there is any missing value is in our data set:
data.isnull().sum()
# we can see there are some missing values in our dataset 
# so we will fill our dataset
data.fillna(' ')
#now there are no missing values

#now we will drop irrelevent features from our featurs column

data = data.drop('Car_Name', axis=1)

#now we will change the categorical values into numerical:

data.replace({'Fuel_Type': {'Diesel': 0, 'Petrol': 1, 'CNG': 2}}, inplace=True)
data.replace({'Seller_Type': {'Dealer': 0, 'Indivisual': 1}}, inplace= True)
data.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace= True)


#now we will seperate our feature and target
x =  data.drop('Selling_price', axis=1)
y = data['Selling_Price']

# now we will send our data into training and testing:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=2)

#now we will call our model:
 
model = LinearRegression()

model.fit(x_train, y_train)

prediction_of_x_train = model.predict(x_train)

#now we will check the accuracy of our model:
metrics.r2_score(prediction_of_x_train, y_train)

#now we will make our predictive system:

user_input = (2014, 5.59,27000, 0, 0, 0,0)
user_input_into_numpy = np.asarray(user_input)
reshaped_array = user_input_into_numpy.reshape(1, -1)
prediction = model.predict(reshaped_array)
print(prediction)