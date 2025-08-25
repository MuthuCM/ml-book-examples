# Example 8.2
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv ("working_hours.csv")
# Convert categorical variables to numerical variables
df_final = pd.get_dummies (df)

# Drop the duplicate 'gender_Male' column if it exists
if 'gender_Male ' in df_final.columns:
    df_final.drop('gender_Male ', axis=1, inplace=True)


# Define Independent and Dependent Variables
Y = df_final ["workHrs"].values
X = df_final.drop ("workHrs", axis=1).values
# Do Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler( )
X = sc.fit_transform(X)

# # Fit Decision Tree Regressor Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor (random_state = 0)
regressor.fit(X,Y)

# Calculate Accuracy
from sklearn.metrics import r2_score
y_pred = regressor.predict(X)
print(y_pred)
print (r2_score (Y, y_pred))

# Do prediction with Decision Tree Regressor Model
testInput       = {'dress' : ["Formal", "BCasual", "Casual"],
           'gender'    : ["male", "female", "male"]
          }
df1     = pd.DataFrame (testInput)
df1_final       = pd.get_dummies (df1)

# Ensure test data has the same columns as training data
# Get the columns from the training data features (excluding the target)
training_columns = df_final.drop('workHrs', axis=1).columns
# Reindex the test data to match the training data columns, filling missing columns with 0
X1_processed = df1_final.reindex(columns=training_columns, fill_value=0)

X1      = X1_processed.values
X1      = sc.transform (X1)
predictedValue  = regressor.predict (X1)
print(predictedValue)
