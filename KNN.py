import pandas as pd 
import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.spatial import distance

print("Hello, World!")

data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])

print(df)

print(distance.euclidean([1, 0, 0], [0, 1, 0]))
sns.barplot(x = "reports", y = "coverage", data=df)

plt.show()

def find_nearest_centroid(df: pd.DataFrame, centroids):
	prev_centroids = [np.random.choice]
	df_copy = df.copy()

	i = 0

