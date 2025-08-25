# Example 8.2
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv ("working_hours.csv")

# Convert categorical variables to numerical variables
df_final = pd.get_dummies (df)

# Define Independent and Dependent Variables
Y = df_final ["workHrs"].values
X = df_final.drop ("workHrs", axis=1).values
# Do Scaling of Data
from sklearn.preprocessing import standardScaler
sc = StandardScaler( )
X = sc.fit_transform(X)

# # Fit Decision Tree Classifier Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor (random_state = 0)
regressor.fit(X,Y)

# Calculate Accuracy
from sklearn.metrics import r2_score
y_pred = regressor.predict(X)
print(y_pred)
print (r2_score (Y, y_pred))

# Do prediction with Decision Tree Classifier Model
testInput	= {'dress' : ["Formal", "BCasual", "Casual"],
	   'gender'    : ["male", "female", "male"]
	  }		 
df1	= pd.DataFrame (testInput)
df1_final	= pd.get_dummies (df1)
X1	= df1_final.iloc[:, :].values
X1	= sc.fit_transform (X1)
predictedValue	= regressor.predict (X1)
print(predictedValue)
