# Example 9.6
# Load Packages
# Example 9.6
# Load Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv('crop_yield.csv')

# Identify categorical columns for encoding
categorical_columns = ['Crop']
label_encoders = {}
for col in categorical_columns:
   le = LabelEncoder()
   data[col] = le.fit_transform(data[col])
# Define Independent and Dependent Variables
target = data['Yield']
      features=data[['Crop','Area','Production','Annual_Rainfall',                               
                                              'Fertilizer', 'Pesticide' ]]       


