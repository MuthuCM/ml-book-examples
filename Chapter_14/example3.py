# Example 14.3
# K-Means Clustering
# Import the libraries
import numpy as np
import pandas as np
import matplotlib.pyplot as plt

df = pd.read_csv("DriverData.csv")
X = df.iloc[:, [2,3]].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler( )
X = sc.fit_transform(X)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100,
                               c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100,
                               c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100,
                               c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_
               [:, 1], s=300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Drivers')
plt.xlabel('Concentration Score (Score1)')
plt.ylabel('Response Score (Score2)')
plt.legend( )
plt.show( )
