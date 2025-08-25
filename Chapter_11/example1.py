# Example 11.1
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
file = "processed.cleveland.data"
# The names will be the names of each column in our pandas DataFrame
names = ['age','sex','cp','trestbps','chol','fbs','restecg', 'thalach',
         'exang', 'oldpeak',   'slope',  'ca',   'thal',  'class']
df = pd.read_csv(file, names=names)

# Remove missing data (indicated with a "?")
df = df[~df.isin(['?'])]
# Drop rows with NaN values from DataFrame
df = df.dropna(axis=0)
df = df.apply(pd.to_numeric)
# Define Independent and Dependent Variables
X = df.drop(['class'], axis=1).values
y = df['class'].values

# Do Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X  = sc.fit_transform(X)

# Fit MLP Classifier Model
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X, y)
# Calculate Accuracy
from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X)
print(accuracy_score(y, y_pred))

# Make Prediction for a new Patient
inputdata = np.array([[63.0,1.0,1.0,145.0,233.0,1.0,2.0,
                                      150.0,0.0,2.3,3.0,0.0,6.0]])
inputdata = inputdata.reshape(len(inputdata), -1)
predictedValue = classifier.predict(inputdata)

print(predictedValue)
