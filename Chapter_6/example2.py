# Example 6.2
# Load Packages
import matplotlib.pyplot as plt
import mglearn

# Load Data
from sklearn.neighbors import KNeighborsClassifier
X, y = mglearn.datasets.make_forge()

# Fit KNN Classifier Model
classifier = KNeighborsClassifier(n_neighbors = 3)
cr = classifier.fit(X,y)

# Calculate Accuracy
from sklearn import metrics
y_pred = classifier.predict(X)
print (metrics.f1_score(y, y_pred, average='weighted'))

# Visualize the Line of Seperation
mglearn.plots.plot_2d_separator(cr, X, fill = False, eps = 0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.title("KNN Classifier")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
mglearn.plots.plot_2d_separator(cr, X, fill = False, eps = 0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.title("KNN Classifier")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()
