# Example 4.1
# Simple Classification using Logistic Regression
# Step 1: Import the Libraries
import numpy as np
# Step 2: Specify Data
salary = [45000, 40000, 35000, 30000, 42000, 37000, 43000, 38000,
          41000, 44000, 90000, 80000, 70000, 60000, 95000,
          85000, 75000, 65000, 84000, 92000]
vehicle_type = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
salary = np.array(salary)
salary = salary.reshape(-1,1)
# Step 3: Transform the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
salary = sc.fit_transform(salary)
# Step 4: Fit Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(salary, vehicle_type)
# Step 5: Calculate Accuracy
from sklearn import metrics
predicted_values = classifier.predict(salary)
print("F-Score: ", metrics.f1_score(vehicle_type, predicted_values,
      average = 'weighted'))
# Step 6: Doing Prediction
X = [75000,92000,31000]
X = np.array(X)
X = X.reshape(-1,1)
X = sc.transform(X)
predictedValues = classifier.predict(X)
print(predictedValues)
