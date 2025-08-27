# Example 9.1
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
names = ['id', 'clump_thickness', 'uniform_cell_size',
        'uniform_cell_shape', 'marginal_adhesion',
        'signle_epithelial_size', 'bare_nuclei', 'bland_chromatin',
        'normal_nucleoli', 'mitoses', 'class']
df = pd.read_table ('breast-cancer-wisconsin.data', sep=',',
                                                       names=names)

# Preprocess the Data
df.replace ('?', -99999, inplace=True)
df.drop (['id'], axis=1, inplace = True)

# Define Independent and Dependent Variables
X = df.iloc [:, [0,1,2,3,4,5,6,7,8]].values
Y = df.iloc [:, 9].values

# Split into Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Do Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler( )
X_train = sc.fit_transform (x_train)
X_test = sc.transform (x_test)

# Fit Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier (n_estimators=10,
                                           criterion='entropy', random_state=0)
classifier.fit (X_train,y_train)

# Calculate Accuracy
from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
print(f"Accuracy Score is: {accuracy_score(y_test, y_pred)}")

# Do prediction with Random Forest Classifier Model
input_data = np.array ([[4,2,1,1,1,2,3,2,1]])
input_data = input_data.reshape(len(input_data),-1)
predicted_value = classifier.predict(input_data)

print (f"Predicted Value is: {predicted_value}")
