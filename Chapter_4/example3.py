# Example 4.3
# Load Packages
import numpy as np
import pandas as pd

# Load Data
df = pd.read_csv ("Health_Data.csv")
X = df.iloc [:, [2, 3]].values
y = df.iloc [:, 4].values

# Fit Logistic Regression Model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler( )
X = sc.fit_transform(X)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression( )
classifier.fit(X,y)

# Calculate Accuracy
#from sklearn.metrics import fl_score
y_pred = classifier.predict(X)
print(sklearn.metrics.f1_score(y, y_pred, average='weighted'))

# Do prediction with Logistic Regression Model
testInput = {"Height":[174, 189, 185], "Weight":[96, 87, 110]}
testData = pd.DataFrame(testInput)
X = testData.iloc [:,:].values
X = sc.fit_transform (X)
predictedValues = classifier.predict(X)
print(predictedValues)
print (predictedValues)
