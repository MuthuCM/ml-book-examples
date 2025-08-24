# Import the packages
Import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Dataset
df = pd.read_csv ("insurance.csv")
df-final = pd.get-dummies (df)

# Define independent variables and dependent variable
y=df_final ["expenses"].values
x=df_final.drop ("expenses", axis=1).values

# Fitting the Model & Calculating Accuracy
from sklearn.linear_model import LinearRegression
regressor = LinearRegression( )
regressor.fit(X,Y)
Y_bat = regressor.predict(X)
print (r2_score(y, y_hat)) #0.75

# Fitting the Model using OLS method
import statsmodels.formula.api as smf
results = smf.ols ('expenses ~ age + sex_male + bmi + children +  smoker_yes + region_northwest + region_southeast + region_southwest', data = df_final).fit( )
print(results.summary)

# Adding Additional Terms
df_final ['age2'] = df_final.age ** 2
df_final ['bmi30'] = (df_final ['bmi'] >=30)*1
df_final ['bmi30_smoker'] = (df_final['bmi'] >=30)* df_final.smoker_yes

# Defining Independent Variables & Dependent Variable again
X = df_final [['age', 'age2', 'children', 'bmi', 'sex_male',
			   'bmi_30', 'smoker_yes', 'region_northwest',
			   'region_southeast', 'region_southwest', 
			   'bmi30_smoker']].values
Y = df_final ["expenses"].values

# Fitting Regression again & Checking Accuracy 
regressor = LinearRegression( )
regressor.fit (X,Y)
y_hat = regressor.predict (X)
print(r2_score (Y,Y_hat)) # 0.87