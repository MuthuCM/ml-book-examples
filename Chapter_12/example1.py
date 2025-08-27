# Example 12.1
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv ("companyData.csv")

# Convert categorical variables to numerical variables
df_final = pd.get_dummies(df)

# Define Independent and Dependent Variables
X = df_final.drop ('Profit', axis = 1).values
Y = df_final.Profit.values

# Split into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X,Y,test_size = 0.2, random_state = 0)

# Fit various Regressor Models
from sklearn.linear_model       import LinearRegression
from sklearn.tree       import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm        import SVR
from sklearn.ensemble   import RandomForestRegressor
from sklearn.ensemble       import ExtraTreesRegressor
from sklearn.ensemble       import AdaBoostRegressor
from sklearn.ensemble       import GradientBoostingRegressor
from sklearn.neural_network     import MLPRegressor
regressor1      =       LinearRegression( )
regressor2      =       DecisionTreeRegressor( )
regressor3      =       KNeighborsRegressor( )
regressor4      =       SVR( )
regressor5      =       RandomForestRegressor( )
regressor6      =       ExtraTreesRegressor( )
regressor7      =       AdaBoostRegressor( )
regressor8      =       GradientBoostingRegressor( )
regressor9      =       MLPRegressor( )
regressor1.fit (X_train, Y_train)
regressor2.fit (X_train, Y_train)
regressor3.fit (X_train, Y_train)
regressor4.fit (X_train, Y_train)
regressor5.fit (X_train, Y_train)
regressor6.fit (X_train, Y_train)
regressor7.fit (X_train, Y_train)
regressor8.fit (X_train, Y_train)
regressor9.fit (X_train, Y_train)

# Calculate Accuracy
from sklearn.metrics import r2_score
print (r2_score (Y_test, regressor1.predict(X_test)))
print (r2_score (Y_test, regressor2.predict(X_test)))
print (r2_score (Y_test, regressor3.predict(X_test)))
print (r2_score (Y_test, regressor4.predict(X_test)))
print (r2_score (Y_test, regressor5.predict(X_test)))
print (r2_score (Y_test, regressor6.predict(X_test)))
print (r2_score (Y_test, regressor7.predict(X_test)))
print (r2_score (Y_test, regressor8.predict(X_test)))
print (r2_score (Y_test, regressor9.predict(X_test)))

# Do Prediction
dictionary1 = {'R&D Spend':[145000], 'Administration':[120000],
                   'Marketing Spend':[385000], 'State':['California']
                          }
df1 = pd.DataFrame (dictionary1)
df1_final = pd.get_dummies (df1)
X_New = df1_final.iloc[:,:].values
predictedValue6 = regressor6.predict(X_New)
print (predictedValue6)
