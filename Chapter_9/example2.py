# Example 9.2
# Load Packages
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_moons

# Load Data
X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 0)

# Split into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                    stratify = y,random_state = 1 )

# Fit Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, random_state = 2)
cr = classifier.fit(X_train,y_train)

# Calculate Accuracy
from sklearn import metrics
y_pred = classifier.predict(X_test)
print ("F-Score: ", metrics.f1_score(y_test, y_pred, average='weighted'))

# Visualize the Decision Trees in Random Forest
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(),classifier.estimators_)):
  ax.set_title("Tree {}".format(i))
  mglearn.plots.plot_tree_partition(X_train,y_train,tree, ax = ax)           
                                                       
mglearn.plots.plot_2d_separator(classifier, X_train, fill = True, 
                                       ax = axes[-1, -1], alpha = 0.4)
axes[-1, -1].set_title("Random Forest")

mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
