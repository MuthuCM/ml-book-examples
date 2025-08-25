# Example 9.3
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = sns.load_dataset("diamonds")

# Define numerical and categorical columns
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_cols = ['cut', 'color', 'clarity']

# Separate features and target variable
X = df.drop("price", axis=1)
y = df.price.values

# Scale numerical features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_numerical_scaled = sc.fit_transform(X[numerical_cols])
X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_cols)

# One-hot encode categorical features
X_categorical_encoded = pd.get_dummies(X[categorical_cols])

# Concatenate scaled numerical and one-hot encoded categorical features
X_processed = pd.concat([X_numerical_scaled, X_categorical_encoded], axis=1)

# Update X to the processed features
X = X_processed.values

# Fit Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor( )
regressor.fit (X,y)

# Do prediction with Random Forest Regressor Model
from sklearn.metrics import r2_score
print (r2_score (y,regressor.predict(X)))
dictionary1 = { 'carat' : [0.21,0.25,0.27,0.29,0.22,0.24,0.26,0.28],   
                       'color': ['D','E','F','G','H','I','J','D'], 
                 'clarity' : ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'], 
                 'depth' : [58.0,58.3,59.6,59.9,60.2,60.5,60.8,61.1],
                'table' : [55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0],
                'x' :[3.90,3.93,3.99,4.02,4.05,4.08,4.11,4.14],
                'y':[3.91,3.92,3.98,4.03,4.04,4.09,4.10,4.15],
                'z' :[2.20,2.24,2.28,2.32,2.36,2.40,2.44,2.48],
      'cut':['Ideal','Premium','Good','VeryGood','Fair','Ideal','Premium','Good']}
df1 = pd.DataFrame (dictionary1)
X1 = df1_final.iloc[:, :].values
X1 = sc.fit_transform(X1)
predictedValues = regressor.predict(X1)

print(predictedValues)
