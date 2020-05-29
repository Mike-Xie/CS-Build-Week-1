import pandas as pd 
import numpy as np 
import scipy as sp
import time
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.spatial import distance
from typing import List, Dict, Optional 

start_time = time.time() 
# test data
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])

# create a class 
class K_means:
	def __init__(self, k: int=3, tolerance: float=0.001, max_iterations: int=100):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations

	def fit(self, data) -> None:
		# put centroids in key value pair
		self.centroids: Dict = {}
		# assign some starting ones, non-randomized, easier to test
		for i in range(self.k):
			self.centroids[i] = data[i]
		# outer loop for max iterations
		for i in range(self.max_iterations):
			self.classifications = {}
			for i in range (self.k):
				self.classifications[i] = []
			# make a list of the distances from each item in data to the centroids
			# get indices of the closest item
			# updated classifications dictionary with that k:v pair of index : item
			for item in data:
				distance_list: List = [np.linalg.norm(item-self.centroids[centroid]) for centroid in self.centroids]
				closest: int = distance_list.index(min(distance_list)) 
				self.classifications[closest].append(item)
			# compare old centroids to new centroids and quit
			# method if the change is less than tolerance 

			# create copy of old centroids to compare to new ones
			prev_centroids = dict(self.centroids)
			# update new centroids 
			for classif in self.classifications:
				self.centroids[classif] = np.average(self.classifications[classif], axis=0)
			# code to exit if has converged to within tolerance
			is_stable: bool = True
			for centroid in self.centroids:
				old = prev_centroids[centroid]
				curr = self.centroids[centroid]
				if np.sum((curr - old)/old*100) > self.tolerance:
					is_stable = False 
			if is_stable:
				break 

	def predict(self, item) -> int:
		distance_list: List = [np.linalg.norm(item-self.centroids[centroid]) for centroid in self.centroids]
		closest: int = distance_list.index(min(distance_list)) 
		return closest 



