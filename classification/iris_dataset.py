#importing required modules
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# loading the iris dataset:
df = sns.load_dataset('iris')
df.head()
#result will be:
# sepal_length	sepal_width	 petal_length	petal_width	   species
#   5.1	              3.5	    1.4	           0.2	        setosa
# 	4.9	              3.0	    1.4	           0.2	        setosa
# 	4.7	              3.2	    1.3	           0.2	        setosa
# 	4.6	              3.1	    1.5	           0.2	        setosa
# 	5.0	              3.6	    1.4	           0.2	        setosa

#so there are 
# features: 4
# target: 1 options :3 - ('setosa', 'versicolor', 'virginica')

# now we will check if there is any missing value is present or not:
df.isnull().sum()
#there are no msising value in our datset.

# nwo we will remove our 1 target which is not relevent:
# there are 2 ways:
# 1:
df['species'] = df.drop('setosa',axis=0, inplace= True)
# 2:
df[df['species'] != 'setosa']
# this will drop setosa value from all dataset

#now we will seperate feature and target:
x = df.iloc[:,:-1]
y = df[:,-1]

#now we will send our data into training and testing set:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#now we will call our model:
model = LogisticRegression()

#now we will use gridsearchcv:
param_grid = {'penalty': ['11', '12','elasticnet'], 'C': {1,2,10,20,30,40,50,60,70}}
cv = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5)

#now we will fit our dataset:
cv.fit(x_train, y_train)

#now we will check accuracy of our model:

# using accuracy score:
pred_on_x_test = cv.predict(x_test)
acc_on_x_test = accuracy_score(y_test, pred_on_x_test)

pred_on_x_train = cv.predict(x_train)
acc_on_x_test = accuracy_score(y_train, pred_on_x_train)

# using classification_report:

acc_on_x_test_class = classification_report(y_test, pred_on_x_test)
acc_on_x_train_class = classification_report(y_train, pred_on_x_train)
