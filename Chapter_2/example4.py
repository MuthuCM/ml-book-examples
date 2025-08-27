# Example 2.4
# Multiple Regression - Prediction of Cut_off Mark
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data from a CSV file
df = pd.read_csv('Mark_Data.csv')
X = df.iloc[:,1:4].values
Y = df.iloc[:,4].values
# Fit Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)
print(f"Regression Coefficients are: {regressor.coef_}")
print()
# Calculate Accuracy Score
from sklearn.metrics import r2_score
Y_pred = regressor.predict(X)
print(f"Accuracy is: {r2_score(Y, Y_pred):5.2f}")
print()
# Do Prediction
dictionary1 = {'Maths' : [190,180,200], 'Physics' : [180,170,170],
               'Chemistry': [180,170,170]
               }
df1 = pd.DataFrame (dictionary1)
X = df1.iloc[:, :].values
predicted_values = regressor.predict(X)
print(f"Predicted Cut-off marks are: {predicted_values}")
