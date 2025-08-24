# Predicting the variety of a flower
# Load Pandas Package
import pandas as pd

# Load Data
df = pd.read_csv('Iris.csv')
df.drop(df.columns[[5, ,6,7,8]], axis=1, inplace = True)
df.head( )
# df.info( )
# df.describe( )

# Do Encoding
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder( )
df['iris'] = labelEncoder.fit_transform(df['iris'])
# df.head( )

# Visualize Correlations
sns.heatmap(df.corr( ), annot = True)
sns.pairplot(df, hue = 'iris', vars =
        ['sepal length', 'sepal width', 'petal length', 'petal width'])

# Define independent and Dependent Variables
X = df.drop(['iris'], axis = 1)
Y = df['iris']

# Fit Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression
classifier.fit(X, y)

# Display the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
y_pred = classifier.predict(X)
cm = confusion_matrix(y, y_pred)
print(cm)
print(classification_report(y,y_pred))

# Make Prediction for a new flower(Setosa)
inputdata = np.array([[5.1, 3.5, 1.4, 0.2]])
inputdata = inputdata.reshape(len(inputdata), -1)
predictedValue = classifier.predict(inputdata)
print(predictedValue)

# Make Prediction for a new flower(Versicolor)
inputdata = np.array([[7.0, 3.2, 4.7, 1.4]])
inputdata = inputdata.reshape(len(inputdata), -1)
predictedValue = classifier.predict(inputdata)
print(predictedValue)

# Make Prediction for a new flower(Virginica)
inputdata = np.array([[6.3, 3.3, 6.0, 2.5]])
inputdata = inputdata.reshape(len(inputdata), -1)
predictedValue = classifier.predict(inputdata)
print(predictedValue)