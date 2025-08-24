# Example 9.4
# Load Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv('waterQuality.csv')

# Replace '#NUM!' with NaN
data = data.replace('#NUM!', pd.NA)
# Drop rows with missing values
data.dropna(inplace=True)
data.head()

# Define Independent and Dependent Variables
X = data.drop('is_safe', axis=1)
y = data['is_safe']

# Split into Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (X, y, test_size=
                                     0.3, random_state = 42)

# Fit AdaBoost Classifier Model
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100, random_state=42 )
model.fit(x_train,y_train)

# Calculate Accuracy
from sklearn.metrics import fl_score
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")