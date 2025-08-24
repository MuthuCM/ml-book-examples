# Example 9.3
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = sns.load_dataset("diamonds")

#  Convert categorical variables to numerical variables
df_final = pd.get_dummies (df)

# Define Independent and Dependent Variables
y = df_final.price.values
X = df_final.drop ("price", axis=1).values

# Do Scaling of Data
from sklearn.preprocessing import StandardScalar
sc = StandardScalar( )
X = sc.fit_transform (X)

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