# Example 14.1
# K-Means Clustering
# Import the Library
import numpy as np
# Specify Data
salary =	[ 45000, 40000, 35000, 30000, 42000, 37000, 43000, 38000,
	  38000, 41000, 44000, 90000, 80000, 70000, 60000, 85000,
	75000, 65000, 84000, 92000, 190000, 180000, 170000,
	160000, 195000, 185000, 175000, 165000, 184000, 192000]
car_type	=	[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
salary = np.array(salary)
salary = salary.reshape(-1,1)
# Transform Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
salary = sc.fit_transform(salary)

# Apply K-Means Technique to the given dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                                           random_state = 0)
y_kmeans = kmeans.fit_predict(salary)

print(y_kmeans)
