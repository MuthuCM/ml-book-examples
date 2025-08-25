# Example 2.6
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load Data
df = pd.read_csv ("Position_Salaries.csv")
X = df.iloc [:, 1:2].values
Y = df.iloc [:, 2].values
# Fit Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression( ) 
lr.fit(X,Y)
# Fit Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 4)
X1 = pr.fit_transform(X)
pr.fit(X1,Y)
lr_2 = LinearRegression()
lr_2.fit(X1,Y)
# Visualize the Polynomial Regression relationship
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot (X_grid, lr_2.predict(pr.fit_transform(X_grid)), color = 'blue')
plt.title ("Polynomial regression") 
plt.xlabel ("Position level")
plt.ylabel ("Salary")
plt.show( )
# Do prediction with Polynomial Regression Model
lr_2.predict(pr.fit_transform([[6.5]]))

