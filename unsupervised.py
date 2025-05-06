'''
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# Sample data: Customer spending habits (income, spending score)
data = np.array([[15, 39], [16, 81], [17, 6], [18, 77], [19, 40], [20, 76]])
# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)
# Cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_
# Visualize clusters
for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], c=('red' if labels[i] == 0 else 'blue'))
plt.scatter(centers[:, 0], centers[:, 1], c='green', marker='X')  # Centroids
plt.title("Customer Clusters")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()
'''

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# Sample data: Income vs. Spending Score
data = np.array([[15, 39], [16, 81], [17, 6], [18, 77], [19, 40], [20, 76],
                 [25, 35], [24, 60], [23, 50], [30, 70]])
# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)
# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_
# Visualize clusters
for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], c=('red' if labels[i] == 0 else 'blue'))
plt.scatter(centers[:, 0], centers[:, 1], c='green', marker='X', s=200, label="Centroids")
plt.title("Customer Segmentation")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
