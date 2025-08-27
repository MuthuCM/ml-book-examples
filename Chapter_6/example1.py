# Example 6.1
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("Kyphosis.csv")

# Do Label Encoding
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder( )
df ['kyphosis'] = labelEncoder.fit_transform (df['Kyphosis'])

# Define Independent and Dependent Variables
X = df.iloc [: , [0, 1, 2]].values
Y = df.iloc [:, 3].values

# Splitting into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X,Y,test_size=0.3)

# Fit KNN Classifier Model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier( )
classifier.fit (X_train,Y_train)

# Visualize Training Accuracy and Test Accuracy
Y_Pred = classifier.predict (X_test)
training_accuracy =  []
test_accuracy = []
neighbors_range = range(1,11)
for n_neighbors in neighbors_range:
   cr = KNeighborsClassifier(n_neighbors = n_neighbors)
   cr.fit(X_train, Y_train)
   training_accuracy.append(cr.score(X_train, Y_train))
   test_accuracy.append(cr.score(X_test, Y_test))
plt.plot(neighbors_range, training_accuracy, label="Training Accuracy")
plt.plot(neighbors_range, test_accuracy, label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

input_data = np.array ([[61, 2, 17]])
input_data = input_data.reshape (len (input_data), -1)
predicted_value = classifier.predict (input_data)
print (predicted_value)
