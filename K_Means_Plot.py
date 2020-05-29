from k_means import K_means
import numpy as np 
import time
import matplotlib.pyplot as plt 

start_time = time.time()
X = np.array([[4,25],[24,94],[31,57],[2,62],[3,70]])

# instantiate and fit 
clf = K_means(k=3)
clf.fit(X)
print(type(clf.centroids))
print(clf.centroids)

# plot centroids
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

# clean up later, should be a 1 or 2 liner in Seaborn
colors = ["g","r","b"]

# plot classifications of points around centroids 
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
 

plt.show()
end_time = time.time()
print (f"runtime: {end_time - start_time} seconds")
