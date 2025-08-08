import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error

data = {
    'Age': [25, 30, np.nan, 45, 35, 29, np.nan, 40, 45, 56, 32, 76, np.nan, 21, 45],
    'Gender': ['Male', 'Female', 'Female', np.nan, 'Male', 'Male', 'Female', 'Female','Female', 'Female','Male', 'Female',  'Male', np.nan, 'Female'],
    'Income': [50000, 60000, 55000, 80000, np.nan, 52000, 58000, 62000, 40000, np.nan, 65000, 54000, 72000, np.nan, 56000],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai', np.nan, 'Delhi', 'Mumbai', 'Delhi', np.nan, 'Mumbai', 'Banglore', 'Mumbai', 'Chennai', 'Delhi'],
    'Spending_Score': [60, 70, 65, 90, 85, 55, 60, np.nan, 76, 80, 43, 60, 80, 65, np.nan]
}

#converting these dataset into dataframe:
df = pd.DataFrame(data)

#our target is 'Spending Score' so we will gonna remove the null values:
df = df.dropna(subset=['Spending_Score'])

# now we will gonna split the feature and target:

x = df.dropna('Spending_Score', axis= 1)
y = df['Spending_Score']

# now we will send it to train and test split:
x_train, y_train, x_test, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

#now we will categorize the data :
num_values = ['Age', 'Income']
str_values = ['Gender', 'City']


num_transformer = Pipeline(
    steps=[
      ('imp', SimpleImputer(strategy= 'mean')),
      ('scaler', StandardScaler())
    ])

str_transformer = Pipeline(
    steps=[
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown= 'ignore'))
    ]
)

preprocesser = ColumnTransformer(
    ('num', num_transformer, num_values),
    ('str', str_transformer, str_values)
)

model = Pipeline(
    ('preprocessor', preprocesser),
    ('model', LinearRegression())
)

model.fit(x_train, y_train)


pr_on_training_data = model.predic(x_train)
pr_on_test_data = model.predict(x_test)


#checking accuracy of our model:
acc_on_training_data = r2_score(pr_on_training_data, y_train)
print(acc_on_training_data)

m_a_e = mean_absolute_error(acc_on_training_data, y_train)
print(m_a_e)