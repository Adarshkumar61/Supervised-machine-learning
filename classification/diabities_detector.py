import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('C:\Users\adars\Adarsh\data.csv')

x = data.drop(columns= 'outcome', axis= 1)
y = data['outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify= y, random_state=2)

linear_model = LogisticRegression()

linear_model.fit(x_train, y_train)

predict = linear_model.predict(x_train)
accurcy_of_x_train = accuracy_score(x_train, predict)

#creating a input parameter:

user_input = ([]) #values
in_numpy = np.asarray(user_input)

reshaped_model = in_numpy.reshape(1, -1)
print(reshaped_model)
