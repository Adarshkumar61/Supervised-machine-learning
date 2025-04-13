import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

#loading the dataset:
data = pd.read_csv('csv_files/Train (1).csv')
#Features are : Item_Identifier	Item_Weight	Item_Fat_Content	Item_Visibility	Item_Type	Item_MRP	Outlet_Identifier	Outlet_Establishment_Year	Outlet_Size	Outlet_Location_Type	Outlet_Type	

# Target is : Item_Outlet_Sales

#checking if there is any missing value:
data.isnull().sum()
#yes there are some missing values in column : Item_Weight(numrical value) and Outlet_Size(categorical value)
#so we will fill values which is their previous one

data.fillna(method='ffill', inplace= True) #it will fill those missing values with their previous one

# now we will drop those columns which are not relevent:

data.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace= True)

#now we will Seperate our Feature and target:
# we denote x : feature
        #   y : target

x = data.drop('Item_Outlet_Sales', axis=1)
y = data['Item_Outlet_Sales']
        
# now we will Seperate our categorical features and numerical features:

categorical_col = x.select_dtypes(include='object').columns
numerical_cols =  x.select_dtypes(exclude='object').columns

#now we convert those alphabetical value into numerical value using column transformer and one hot encoder:

converter = ColumnTransformer(
    transformers= [
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_col),
        ('num', 'passthrough', numerical_cols)
    ]
)