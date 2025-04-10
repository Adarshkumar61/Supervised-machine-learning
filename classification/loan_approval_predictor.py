#importing reuqired modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#loading our loan_datset
data  = pd.read_csv('csv_files/train_u6lujuX_CVtuZ9i (1) (1).csv')

#features are: loan_Id, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
#Target : Loan_Status

# now we will drop unrelevent feature from our dataset:

data = data.drop('Loan_ID', axis=1)

#now we will check if there is any value is missing or not
data.isnull().sum()
#we can see there are some missing values in our datset
#so we will remove missing values
#if we want to remove some specific rows value we will use
# data = data.dropna(subset=['LoanAmount'])

# or if we want to remove all missing values:

data  = data.dropna()

#now we will fill those missing values:
data = data.fillna(' ')

# now we will seperate the features and data

x = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

#now we will seperate featues and  with their dtype

categorical_value = x.select_dtypes(include='object').columns.tolist()
numerical_value = x.select_dtypes(exclude='object').columns.tolist()

joiner = ColumnTransformer(
    transformers=[
        ('onehotencode', OneHotEncoder(handle_unknown='ignore'), categorical_value),
        ('standardscaler', StandardScaler(), numerical_value)
    ]
)

model = Pipeline(
    steps=[
        ('joiner', joiner),
        ('logistic', LogisticRegression())
    ]
)

# Splitting data for training and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
model.fit(x_train, y_train)  # Fit the pipeline to the training data
pred = model.predict(x_train)

print(model.score(x_train, y_train))  # Evaluate the model on the training data

#now we will create a predictive system for users:

user_input = {
    'Gender': input('Enter your gender here(male/female): '.title()),
    'Married': input('are you married(yes/no): '.title()),
    'Dependents': float(input('1/2/3 or more: ')),
    'Education': input('are you graduate or not(yes/no): '.title()),
    'Self_Employed': input('are you working(yes/no): '.title()),
    'ApplicantIncome': float(input('enter your income of 1 month: ')),
    'CoapplicantIncome': float(input('enter your coapplicant salary of 1 month: ')),
    'LoanAmount': float(input('enter your loan amount you want(in dollars $): ')),
    'Loan_Amount_Term': float(input('enter for how many moths you want (120: 1year, 240: 2 year, 360: 3 year(max))choose between: (120, 240, 360): ')),
    'Credit_History': float(input('enter your credit history (max: 1.0): ')),
    'Property_Area': input('enter your property area (rural/urban): '.title())
}
user = pd.DataFrame([user_input])
prediction= model.predict(user)
print(prediction)
if prediction == [0]:
    print('congratulation your loan has been approved'.title())
else:
    print('sorry! due to some reason your loan is not approved..'.title())