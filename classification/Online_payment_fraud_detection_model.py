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

