import pandas as pd
import matplotlib.pyplot as plt


data = {
    'Age': [25, 30, np.nan, 45, 35, 29, np.nan, 40, 45, 56, 32, 76, np.nan, 21, 45],
    'Gender': ['Male', 'Female', 'Female', np.nan, 'Male', 'Male', 'Female', 'Female','Female', 'Female','Male', 'Female',  'Male', np.nan, 'Female'],
    'Income': [50000, 60000, 55000, 80000, np.nan, 52000, 58000, 62000, 40000, np.nan, 65000, 54000, 72000, np.nan, 56000],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai', np.nan, 'Delhi', 'Mumbai', 'Delhi', np.nan, 'Mumbai', 'Banglore', 'Mumbai', 'Chennai', 'Delhi'],
    'Spending_Score': [60, 70, 65, 90, 85, 55, 60, np.nan, 76, 80, 43, 60, 80, 65, np.nan]
}