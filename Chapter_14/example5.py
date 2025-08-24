# Example 14.5
# K-Means Clustering
# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplotas plt

df = pd.read_csv ("Mall_Customers.csv")
X = df.iloc[:, [3,4]].values
from sklearn.preprocessing import StandardScalar
sc = StandardScalar( )
X = sc.fit_transform(X)

import scipy.cluster.hierarchy as sch
dendrogram =sch.dendrogram (sch.linkage (X, method='ward'))
plt.title ('Dendrogram')
plt.xlabel ('Customers')
plt.ylabel ('Euclidean Distances')
plt.show ( )