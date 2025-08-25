# Example 7.1
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("Health_Risk_Data.csv")

# Do Label Encoding
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
df["RiskLevel"] = labelEncoder.fit_transform(df["RiskLevel"])

# Define Independent and Dependent Variables
X = df.drop(['RiskLevel'], axis = 1).values
Y = df['RiskLevel'].values

# Do Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler( )
X = sc.fit_transform (X)

# Fit KNN Classifier Model
from sklearn.svm import SVC
classifier = SVC (kernel='linear', random_state=0)
classifier.fit (X,Y)

# Calculate Accuracy
from sklearn import metrics
y_predict = classifier.predict(X)
print ("F-Score: ", metrics.fl_score(Y, y_predict, average = 'weighted'))

# Do Prediction
input_data = np.array ([[35,120,60,6.1,98.0,76]])
input_data = input_data.reshape (len(input_data), -1)
input_data  = sc.fit_transform(input_data )
predicted_value = classifier.predict (input_data)

print (predicted_value)
