# Example 12.2
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv ("customerData.csv")

# Define Independent and Dependent Variables
X = df.iloc[:, [2,3]].values
Y = df.iloc[:, 4].values

# Do Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler( )
X = sc.fit_transform(X)

# Split into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Fit various Classifier Models
from sklearn.linear_model       import LogisticRegression
from sklearn.tree       import DecisionTreeClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm        import SVC
from sklearn.ensemble   import RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
classifier1     =       LogisticRegression( )
classifier2     =       DecisionTreeClassifier( )
classifier3     =       KNeighborsClassifier( )
classifier4     =       SVC( )
classifier5     =       RandomForestClassifier( )
classifier6     =       ExtraTreesClassifier( )
classifier7     =       AdaBoostClassifier( )
classifier8     =       GradientBoostingClassifier( )
classifier9     =       MLPClassifier( )
classifier1.fit(X_train, Y_train)
classifier2.fit(X_train, Y_train)
classifier3.fit(X_train, Y_train)
classifier4.fit(X_train, Y_train)
classifier5.fit(X_train, Y_train)
classifier6.fit(X_train, Y_train)
classifier7.fit(X_train, Y_train)
classifier8.fit(X_train, Y_train)
classifier9.fit(X_train, Y_train)

# Calculate Accuracy
from sklearn.metrics import f1_score
print (f1_score (Y_test, classifier1.predict (X_test)))
print (f1_score (Y_test, classifier2.predict (X_test)))
print (f1_score (Y_test, classifier3.predict (X_test)))
print (f1_score (Y_test, classifier4.predict (X_test)))
print (f1_score (Y_test, classifier5.predict (X_test)))
print (f1_score (Y_test, classifier6.predict (X_test)))
print (f1_score (Y_test, classifier7.predict (X_test)))
print (f1_score (Y_test, classifier8.predict (X_test)))
print (f1_score (Y_test, classifier9.predict (X_test)))

# Do Prediction
dictionary1 = { 'Age' : [19,24,29,34,39],
                'Salary' : [19000,24000,29000,34000,45000],
                    }
df1 = pd.DataFrame (dictionary1)
X1 = df1.iloc[:, :].values
X1 = sc.fit_transform(X1)
predictedValue4 = classifier4.predict(X1)
print(predictedValue4)
