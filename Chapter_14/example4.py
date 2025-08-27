# Example 14.4
# K-Means Clustering
# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")

# Replace '?' with NaN and drop rows with NaN
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

X = df.iloc[:, [3,4]].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler( )
X = sc.fit_transform(X)

from sklearn.cluster import KMeans
wcss = [ ]
for i in range (1, 11):
   kmeans = KMeans (n_clusters=i, init='k-means++', random_state=0)
   kmeans.fit(X)
   wcss.append (kmeans.inertia_)
plt.plot (range(1,11), wcss)
plt.title ('The Elbow Method')
plt.xlabel ('Number of Clusters')
plt.ylabel ('WCSS')
plt.show( )

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1],s=100, c='red',
			label='cluster1')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1],s=100, c='blue',
			label='cluster2')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1],s=100, c='green',
			label='cluster3')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1],s=100, c='cyan',
			label='cluster4')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1],s=100, c='magenta',
			label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],
			kmeans.cluster_centers_[:,1], s=300, c='yellow', label='centroids')
plt.title('Clusters of Customers')
plt.xlabel("Annual Income ('000)")
plt.ylabel('SpendingScore(1-100)')
plt.legend( )
plt.show( )
