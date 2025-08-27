# Example 15.1
# Dimensionality Reduction
# Import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('waterQuality.csv')
# Replace '#NUM!' with NaN and drop rows with NaN
dataset = dataset.replace('#NUM!', np.nan)
dataset.dropna(inplace=True)

X = dataset.drop('is_safe', axis=1)
y = dataset['is_safe']
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_standardized,
                            y, test_size=0.2, random_state=42)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# Defining different Classification Models
models = {
   'Logistic Regression': LogisticRegression(random_state=42),
   'Decision Tree': DecisionTreeClassifier(random_state=42),
   'Random Forest': RandomForestClassifier(random_state=42),
   'SVM': SVC(random_state=42)}

# Function to train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test,
                                               model, method_name):
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f"{method_name} Accuracy: {accuracy:.2f}")
# Loop through each model
for model_name, model in models.items():
   print(f"\n{model_name}:\n")
   # Classification after PCA
   train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test, model,   f"{model_name} after PCA")
