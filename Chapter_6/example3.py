# Example 6.3
# Load Packages
import mumpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv ("usedcars.csv")

# Convert Categorical data into numeric data
df_final = pd.get_dummies(df)

# Define Independent and Dependent Variables
Y = df_final ["price"].values
X = df_final.drop ("price", axis=1).values

# Do Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler
X = sc.fit_transform(X)

# Fit KNN Classifier Model
from sklearn.neighbors import KNeighborsRegressor
regressor = KneighborsRegressor( )
regressor.fit (X,Y)

# Calculate Accuracy
from sklearn.metrics import r2-score
print (r2_score (Y, regressor.predict(x)))