# Example 7.3
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mglearn
from sklearn.svm import SVR

# Load Data
df = sns.load_dataset("tips")

# Convert Categorical Variables to Numeric Variables
df_final = pd.get_dummies(df)

# Define Independent and Dependent Variables
X = df_final.drop('tip', axis =1).values
y = df_final.tip.values

# Do Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# Fit SVR Regressor Model
regressor = SVR()
regressor.fit(X,y)

# Calculate Accuracy
from sklearn.metrics import r2_score
y_pred = regressor.predict(X)
accuracy_score = r2_score(y, y_pred)

# Do prediction with SVR Regressor Model
dictionary1 = {'total_bill':[16.99, 10.34, 21.01, 23.68],
               'sex':[ 'Female', 'Male', 'Female', 'Male'],
               'smoker':[ 'No', 'Yes', 'No', 'Yes'],
	'day':['Thur', 'Fri', 'Sat','Sun'],
               'time':['Lunch','Dinner', 'Lunch', 'Dinner' ],
               'size':[2, 2, 4, 3]
               }			 
df1 = pd.DataFrame(dictionary1)
df1_final = pd.get_dummies(df1)
X1 = df1_final.iloc [:,:].values
X1 = sc.fit_transform (X1)
predictedValue = regressor.predict (X1)

print(predictedValue)
