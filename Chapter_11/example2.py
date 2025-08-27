# Example 11.2
# Load Packages
import numpy as np
import pandas as pd

# Load Data
file = "Autism_Adult_Data.txt"
data = pd.read_table(file, sep = ',', index_col= None)

# Drop unnecessary columns
data = data.drop(['result', 'age_desc'], axis=1)

# Define Independent and Dependent Variables
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
X = pd.get_dummies(x)
X.columns.values
Y = pd.get_dummies(y)

# Fit MLP Classifier Model
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X, Y)

# Calculate Accuracy
from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X)
print(accuracy_score(Y, y_pred))
