# Example 2.5
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv ("polynomial_values.csv")
X = df.iloc [:, 0].values.reshape(-1, 1)
Y = df.iloc [:, 1].values

# Fit Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression( )
lr.fit(X,Y)

# Visualize the Polynomial Regression relationship
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot (X_grid, lr_2.predict(pr.fit_transform(X_grid)), color = 'blue')
plt.title ("Polynomial regression") 
plt.xlabel ("Position level")
plt.ylabel ("Salary")
plt.show( )
