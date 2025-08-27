# Example 7.2
# Load Packages
import matplotlib.pyplot as plt
import mglearn
from sklearn.svm import SVC

# Load Data
X, y = mglearn.datasets.make_forge()

# Fit KNN Classifier Model
classifier = SVC()
cr = classifier.fit(X,y)

# Calculate Accuracy
from sklearn import metrics
y_pred = classifier.predict(X)
print ("F-Score: ", metrics.f1_score(y, y_pred, average='weighted'))

# Visualize the line of separation
mglearn.plots.plot_2d_separator(cr, X, fill = False, eps = 0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.title("SVM Classifier")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
