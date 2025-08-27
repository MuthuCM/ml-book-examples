# Example 8.1
!pip install mglearn
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
dataset = pd.read_csv ("Health_Data.csv")

# Define Independent and Dependent Variables
X = dataset.iloc[:, [2,3]].values
y = df.iloc[:, 4].values

# Do Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler( )
X = sc.fit_transform(X)

# Fit Decision Tree Classifier Model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier( )
classifier.fit (X,Y)

# Calculate Accuracy
from sklearn.metrics import f1_score
y_pred = classifier.predict(X)
print ("F-Score: ", f1_score(y, y_pred, average='weighted'))

# Do Prediction
testInput       = {"Height": [174,189, 185], "Weight" : [96, 87, 110]}
testData        = pd.DataFrame (testInput)
X1              = testData.iloc [:, :].values
X1              = sc.fit_transform (X1)
predictedValue  = classifier.predict (X1)
print (predictedValue)

# Visualize the Decision Tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
feature_cols = ['Height' , 'Weight']
dot_data = StringIO()
export_graphviz(classifier, out_file = "tree.dot",
  filled = True, rounded = True, special_characters = True,
            feature_names = feature_cols,
            class_names=['Under Weight','Normal Weight','Over Weight'])
import graphviz
with open("tree.dot") as f:
   dot_graph = f.read()
   display(graphviz.Source(dot_graph))

