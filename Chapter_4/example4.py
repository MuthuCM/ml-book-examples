# Example 4.4
# Import the Packages
! pip install mglearn
import matplotlib.pyplot as plt
import mglearn
from sklearn.linear_model import Logisticregression

# Load the data
X, y = mglearn.datasets.make_forge()

# Fit the Logistic Regression Model
classifier = LogisticRegression( )
cr = classifier.fit(X,y)
# Calculate Accuracy
from sklearn import metrics
y_pred = classifier.predict(X)
print (metrics.fl_score(y, y_pred, average='weighted'))

# Do Visualization
mglearn.plots.plot_2d_separator(cr, X, fill = False, eps = 0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.title("Logistic Regression Classifier")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()
