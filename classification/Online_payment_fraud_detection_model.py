import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('fraud_data.csv')
print(data.head())

#  step      type    amount     nameOrig  oldbalanceOrg  newbalanceOrig  \
# 0     1   PAYMENT   9839.64  C1231006815       170136.0       160296.36   
# 1     1   PAYMENT   1864.28  C1666544295        21249.0        19384.72   
# 2     1  TRANSFER    181.00  C1305486145          181.0            0.00   
# 3     1  CASH_OUT    181.00   C840083671          181.0            0.00   
# 4     1   PAYMENT  11668.14  C2048537720        41554.0        29885.86   

#       nameDest  oldbalanceDest  newbalanceDest  isFraud  isFlaggedFraud  
# 0  M1979787155             0.0             0.0        0               0  
# 1  M2044282225             0.0             0.0        0               0  
# 2   C553264065             0.0             0.0        1               0  
# 3    C38997010         21182.0             0.0        1               0  
# 4  M1230701703             0.0             0.0        0               0  

data.isnull().sum()
# step              0
# type              0
# amount            0
# nameOrig          0
# oldbalanceOrg     0
# newbalanceOrig    0
# nameDest          0
# oldbalanceDest    0
# newbalanceDest    0
# isFraud           0
# isFlaggedFraud    0

# transforming categorical data to numerical:

data['type'] = data['type'].map({'CASH_OUT': 0, 'PAYMENT': 1, 'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4})

data['isFraud'] = data['isFraud'].map({0: "No Fraud", 1: "Fraud"})

x = data['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
y = data['isFraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))
# 0.9995

#predicting a new data point:
#features = [type, amount, oldbalancerg, newbalancerig]
new_data = np.array([[0, 181.00, 181.0, 0.00]])
prediction = model.predict(new_data)